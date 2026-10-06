import base64
import json
import sys
import time
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

import config as base_config
from . import config as mt_config
from . import db as mt_db


def _get_client():
    try:
        from anthropic import Anthropic
    except ImportError:
        raise RuntimeError(
            "The `anthropic` package is required for VLM steps. "
            "Install with: pip install anthropic"
        )
    if not base_config.ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Put it in .env at the repository root, or export it."
        )
    return Anthropic(api_key=base_config.ANTHROPIC_API_KEY)


def _prepare_image_b64(image_path, max_dim=None):
    max_dim = max_dim or base_config.CLAUDE_MAX_IMAGE_DIM
    img = Image.open(image_path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"


def _extract_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if "```" in text:
            text = text[: text.rfind("```")].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    for end in range(len(text), start, -1):
        if text[end - 1] == "}":
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                continue
    raise json.JSONDecodeError("Could not extract valid JSON", text, 0)


_PROMPT_CACHE = {}


def _load_prompt(step_name):
    if step_name not in _PROMPT_CACHE:
        path = mt_config.PROMPTS_DIR / f"{step_name}.md"
        _PROMPT_CACHE[step_name] = path.read_text(encoding="utf-8")
    return _PROMPT_CACHE[step_name]


def _build_system_block():
    return [{"type": "text", "text": _load_prompt("system")}]


def run_size_gate(scope, force=False, threshold=None, db_path=None):
    threshold = threshold or base_config.MIN_IMAGE_DIM_PX
    print(f"\n[size_gate] threshold = {threshold}px")

    conn = mt_db.get_connection(db_path)
    rows = mt_db.select_candidates_for_size_gate(conn, scope, force=force)
    print(f"[size_gate] candidates to evaluate: {len(rows):,}")
    if not rows:
        conn.close()
        return {"evaluated": 0, "pass": 0, "fail_too_small": 0}

    n_pass = 0
    n_fail = 0
    for row in tqdm(rows, desc="size_gate"):
        w = row["image_width"]
        h = row["image_height"]
        if w is None or h is None:
            status = "fail_too_small"
            min_dim = 0
        else:
            min_dim = min(w, h)
            status = "pass" if min_dim >= threshold else "fail_too_small"
        mt_db.upsert_size_gate(conn, row["uid"], status, min_dim)
        if status == "pass":
            n_pass += 1
        else:
            n_fail += 1
    conn.commit()
    conn.close()

    print(f"[size_gate] pass: {n_pass:,}   fail_too_small: {n_fail:,}")
    return {"evaluated": len(rows), "pass": n_pass, "fail_too_small": n_fail}


def _build_step1_user_content(image_b64, media_type, prompt_text):
    return [
        {
            "type": "text",
            "text": prompt_text,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": image_b64},
        },
    ]


def _call_step1(client, model, image_path):
    image_b64, media_type = _prepare_image_b64(image_path)
    prompt_text = _load_prompt("step1")
    system_block = _build_system_block()

    message = client.messages.create(
        model=model,
        max_tokens=mt_config.MAX_TOKENS_PER_STEP["step1"],
        system=system_block,
        messages=[{"role": "user", "content": _build_step1_user_content(image_b64, media_type, prompt_text)}],
    )

    response_text = message.content[0].text.strip()
    parsed = _extract_json(response_text)

    usage = message.usage
    return {
        "parsed": parsed,
        "raw": response_text,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation": getattr(usage, "cache_creation_input_tokens", 0) or 0,
    }


def _validate_step1_response(parsed):
    required = ["is_map", "is_map_dominant", "short_description"]
    for f in required:
        if f not in parsed:
            raise ValueError(f"missing field: {f}")
    if not isinstance(parsed["is_map"], bool):
        raise ValueError(f"is_map must be bool, got {type(parsed['is_map']).__name__}")
    if not isinstance(parsed["is_map_dominant"], bool):
        raise ValueError(f"is_map_dominant must be bool, got {type(parsed['is_map_dominant']).__name__}")
    if not isinstance(parsed["short_description"], str):
        raise ValueError(f"short_description must be str, got {type(parsed['short_description']).__name__}")
    return parsed


def _process_step1_one(client, model, uid, image_path):
    last_error = None
    for attempt in range(mt_config.RETRY_ATTEMPTS):
        try:
            result = _call_step1(client, model, image_path)
            parsed = _validate_step1_response(result["parsed"])
            return {
                "uid": uid,
                "status": "done",
                "model": model,
                "raw_response": result["raw"],
                "is_map": parsed["is_map"],
                "is_map_dominant": parsed["is_map_dominant"],
                "short_description": parsed["short_description"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "cache_read": result["cache_read"],
                "cache_creation": result["cache_creation"],
                "error": None,
            }
        except (json.JSONDecodeError, ValueError) as e:
            last_error = f"parse: {e}"
            if attempt < mt_config.RETRY_ATTEMPTS - 1:
                time.sleep(mt_config.RETRY_DELAY_S)
                continue
            return {
                "uid": uid, "status": "parse_error", "model": model,
                "raw_response": None,
                "is_map": None, "is_map_dominant": None, "short_description": None,
                "input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_creation": 0,
                "error": last_error,
            }
        except Exception as e:
            last_error = str(e)
            txt = last_error.lower()
            if "rate_limit" in txt or "429" in txt or "overloaded" in txt or "529" in txt:
                wait = mt_config.RETRY_DELAY_S * (attempt + 1) * mt_config.RATE_LIMIT_BACKOFF_MULTIPLIER
                time.sleep(wait)
                continue
            if attempt < mt_config.RETRY_ATTEMPTS - 1:
                time.sleep(mt_config.RETRY_DELAY_S)
                continue
            return {
                "uid": uid, "status": "error", "model": model,
                "raw_response": None,
                "is_map": None, "is_map_dominant": None, "short_description": None,
                "input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_creation": 0,
                "error": last_error,
            }
    return {
        "uid": uid, "status": "error", "model": model,
        "raw_response": None,
        "is_map": None, "is_map_dominant": None, "short_description": None,
        "input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_creation": 0,
        "error": last_error or "max retries exhausted",
    }


def run_step1(scope, force=False, model=None, workers=None, db_path=None):
    model = model or mt_config.MODEL_PER_STEP["step1"]
    workers = workers or mt_config.WORKERS

    print(f"\n[step1] model = {model}   workers = {workers}")

    conn = mt_db.get_connection(db_path)
    rows = mt_db.select_candidates_for_step1(conn, scope, force=force)
    print(f"[step1] candidates to evaluate: {len(rows):,}")
    if not rows:
        conn.close()
        return {"evaluated": 0, "done": 0, "errors": 0}

    client = _get_client()

    done = 0
    errors = 0
    parse_errors = 0
    total_in = total_out = total_cr = total_cw = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for row in rows:
            image_path = row["local_path"]
            if not image_path or not Path(image_path).exists():
                mt_db.upsert_step1_result(
                    conn, row["uid"],
                    status="error", model=model, raw_response=None,
                    error="local image file not found",
                )
                errors += 1
                continue
            futures[executor.submit(_process_step1_one, client, model, row["uid"], image_path)] = row["uid"]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="step1"):
            res = fut.result()
            mt_db.upsert_step1_result(
                conn, res["uid"],
                status=res["status"], model=res["model"], raw_response=res["raw_response"],
                is_map=res["is_map"], is_map_dominant=res["is_map_dominant"],
                short_description=res["short_description"],
                input_tokens=res["input_tokens"], output_tokens=res["output_tokens"],
                cache_read=res["cache_read"], cache_creation=res["cache_creation"],
                error=res["error"],
            )
            conn.commit()
            total_in += res["input_tokens"]
            total_out += res["output_tokens"]
            total_cr += res["cache_read"]
            total_cw += res["cache_creation"]
            if res["status"] == "done":
                done += 1
            elif res["status"] == "parse_error":
                parse_errors += 1
            else:
                errors += 1

    conn.close()

    cost = _estimate_cost_haiku_or_sonnet(model, total_in, total_out, total_cr, total_cw)
    print(f"[step1] done: {done:,}   parse_errors: {parse_errors:,}   errors: {errors:,}")
    print(f"[step1] tokens — in: {total_in:,}  out: {total_out:,}  cache_read: {total_cr:,}  cache_write: {total_cw:,}")
    print(f"[step1] estimated cost: ${cost:.4f}")

    return {
        "evaluated": len(rows), "done": done, "parse_errors": parse_errors, "errors": errors,
        "tokens": {"input": total_in, "output": total_out, "cache_read": total_cr, "cache_creation": total_cw},
        "cost_usd": cost,
    }


def _build_step2_messages(image_b64, media_type, step1_prompt, step1_raw_response, step2_prompt):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": step1_prompt,
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": image_b64},
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": step1_raw_response}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": step2_prompt,
                },
            ],
        },
    ]


def _call_step2(client, model, image_path, step1_raw_response):
    image_b64, media_type = _prepare_image_b64(image_path)
    step1_prompt = _load_prompt("step1")
    step2_prompt = _load_prompt("step2")
    system_block = _build_system_block()

    messages = _build_step2_messages(image_b64, media_type, step1_prompt, step1_raw_response, step2_prompt)

    message = client.messages.create(
        model=model,
        max_tokens=mt_config.MAX_TOKENS_PER_STEP["step2"],
        system=system_block,
        messages=messages,
    )

    response_text = message.content[0].text.strip()
    parsed = _extract_json(response_text)

    usage = message.usage
    return {
        "parsed": parsed,
        "raw": response_text,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation": getattr(usage, "cache_creation_input_tokens", 0) or 0,
    }


def _validate_step2_response(parsed):
    required = ["map_language", "is_statistical_map", "has_admin_units", "has_quantitative_data", "has_text"]
    for f in required:
        if f not in parsed:
            raise ValueError(f"missing field: {f}")
    for f in ("is_statistical_map", "has_admin_units", "has_quantitative_data", "has_text"):
        if not isinstance(parsed[f], bool):
            raise ValueError(f"{f} must be bool, got {type(parsed[f]).__name__}")
    if not isinstance(parsed["map_language"], str):
        raise ValueError(f"map_language must be str, got {type(parsed['map_language']).__name__}")
    if not parsed["is_statistical_map"] and parsed["has_quantitative_data"]:
        parsed["has_quantitative_data"] = False
    return parsed


def _process_step2_one(client, model, uid, image_path, step1_raw_response):
    last_error = None
    for attempt in range(mt_config.RETRY_ATTEMPTS):
        try:
            result = _call_step2(client, model, image_path, step1_raw_response)
            parsed = _validate_step2_response(result["parsed"])
            if not parsed["is_statistical_map"]:
                data_level = "not_applicable"
            elif parsed["has_quantitative_data"]:
                data_level = "quantitative"
            else:
                data_level = "non_quantitative"
            return {
                "uid": uid,
                "status": "done",
                "model": model,
                "raw_response": result["raw"],
                "map_language": parsed["map_language"],
                "is_statistical_map": parsed["is_statistical_map"],
                "has_admin_units": parsed["has_admin_units"],
                "data_level": data_level,
                "has_text": parsed["has_text"],
                "has_quantitative_data": parsed["has_quantitative_data"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "cache_read": result["cache_read"],
                "cache_creation": result["cache_creation"],
                "error": None,
            }
        except (json.JSONDecodeError, ValueError) as e:
            last_error = f"parse: {e}"
            if attempt < mt_config.RETRY_ATTEMPTS - 1:
                time.sleep(mt_config.RETRY_DELAY_S)
                continue
            return _step2_failure(uid, model, "parse_error", last_error)
        except Exception as e:
            last_error = str(e)
            txt = last_error.lower()
            if "rate_limit" in txt or "429" in txt or "overloaded" in txt or "529" in txt:
                wait = mt_config.RETRY_DELAY_S * (attempt + 1) * mt_config.RATE_LIMIT_BACKOFF_MULTIPLIER
                time.sleep(wait)
                continue
            if attempt < mt_config.RETRY_ATTEMPTS - 1:
                time.sleep(mt_config.RETRY_DELAY_S)
                continue
            return _step2_failure(uid, model, "error", last_error)
    return _step2_failure(uid, model, "error", last_error or "max retries exhausted")


def _step2_failure(uid, model, status, error):
    return {
        "uid": uid, "status": status, "model": model, "raw_response": None,
        "map_language": None, "is_statistical_map": None, "has_admin_units": None,
        "data_level": None, "has_text": None, "has_quantitative_data": None,
        "input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_creation": 0,
        "error": error,
    }


def run_step2(scope, force=False, model=None, workers=None, db_path=None):
    model = model or mt_config.MODEL_PER_STEP["step2"]
    workers = workers or mt_config.WORKERS

    print(f"\n[step2] model = {model}   workers = {workers}")
    print(f"[step2] gate (Python): mt_is_map=1 AND mt_is_map_dominant=1")

    conn = mt_db.get_connection(db_path)
    rows = mt_db.select_candidates_for_step2(conn, scope, force=force)
    print(f"[step2] candidates to evaluate: {len(rows):,}")
    if not rows:
        conn.close()
        return {"evaluated": 0, "done": 0, "errors": 0}

    client = _get_client()

    done = 0
    errors = 0
    parse_errors = 0
    total_in = total_out = total_cr = total_cw = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for row in rows:
            image_path = row["local_path"]
            step1_raw = row["mt_step1_raw_response"]
            if not image_path or not Path(image_path).exists():
                mt_db.upsert_step2_result(
                    conn, row["uid"],
                    status="error", model=model, raw_response=None,
                    error="local image file not found",
                )
                errors += 1
                continue
            if not step1_raw:
                mt_db.upsert_step2_result(
                    conn, row["uid"],
                    status="error", model=model, raw_response=None,
                    error="mt_step1_raw_response missing — rerun step1 with --force",
                )
                errors += 1
                continue
            futures[executor.submit(
                _process_step2_one, client, model, row["uid"], image_path, step1_raw
            )] = row["uid"]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="step2"):
            res = fut.result()
            mt_db.upsert_step2_result(
                conn, res["uid"],
                status=res["status"], model=res["model"], raw_response=res["raw_response"],
                map_language=res["map_language"],
                is_statistical_map=res["is_statistical_map"],
                has_admin_units=res["has_admin_units"],
                data_level=res["data_level"],
                has_text=res["has_text"],
                has_quantitative_data=res["has_quantitative_data"],
                input_tokens=res["input_tokens"], output_tokens=res["output_tokens"],
                cache_read=res["cache_read"], cache_creation=res["cache_creation"],
                error=res["error"],
            )
            conn.commit()
            total_in += res["input_tokens"]
            total_out += res["output_tokens"]
            total_cr += res["cache_read"]
            total_cw += res["cache_creation"]
            if res["status"] == "done":
                done += 1
            elif res["status"] == "parse_error":
                parse_errors += 1
            else:
                errors += 1

    conn.close()

    cost = _estimate_cost_haiku_or_sonnet(model, total_in, total_out, total_cr, total_cw)
    print(f"[step2] done: {done:,}   parse_errors: {parse_errors:,}   errors: {errors:,}")
    print(f"[step2] tokens — in: {total_in:,}  out: {total_out:,}  cache_read: {total_cr:,}  cache_write: {total_cw:,}")
    print(f"[step2] estimated cost: ${cost:.4f}")

    return {
        "evaluated": len(rows), "done": done, "parse_errors": parse_errors, "errors": errors,
        "tokens": {"input": total_in, "output": total_out, "cache_read": total_cr, "cache_creation": total_cw},
        "cost_usd": cost,
    }


_METHOD_FIELDS = (
    "has_choropleth", "has_diagrams", "has_isolines",
    "has_dot_density", "has_heat_map", "has_cartogram", "has_flow_map",
)
_CONFIDENCE_VALUES = {"low", "medium", "high"}


def _build_step3_messages(image_b64, media_type, step1_prompt, step1_raw_response,
                           step2_prompt, step2_raw_response, step3_prompt):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": step1_prompt,
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": image_b64},
                },
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": step1_raw_response}]},
        {"role": "user", "content": [{"type": "text", "text": step2_prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": step2_raw_response}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": step3_prompt,
                },
            ],
        },
    ]


def _call_step3(client, model, image_path, step1_raw, step2_raw):
    image_b64, media_type = _prepare_image_b64(image_path)
    step1_prompt = _load_prompt("step1")
    step2_prompt = _load_prompt("step2")
    step3_prompt = _load_prompt("step3")
    system_block = _build_system_block()

    messages = _build_step3_messages(
        image_b64, media_type, step1_prompt, step1_raw, step2_prompt, step2_raw, step3_prompt
    )

    message = client.messages.create(
        model=model,
        max_tokens=mt_config.MAX_TOKENS_PER_STEP["step3"],
        system=system_block,
        messages=messages,
    )

    response_text = message.content[0].text.strip()
    parsed = _extract_json(response_text)
    usage = message.usage
    return {
        "parsed": parsed,
        "raw": response_text,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation": getattr(usage, "cache_creation_input_tokens", 0) or 0,
    }


def _validate_step3_response(parsed):
    for f in _METHOD_FIELDS:
        if f not in parsed:
            raise ValueError(f"missing method field: {f}")
        if not isinstance(parsed[f], bool):
            raise ValueError(f"{f} must be bool, got {type(parsed[f]).__name__}")
    if "confidence" not in parsed:
        raise ValueError("missing field: confidence")
    if parsed["confidence"] not in _CONFIDENCE_VALUES:
        raise ValueError(f"confidence must be one of {_CONFIDENCE_VALUES}, got {parsed['confidence']!r}")
    if "legend_is_classed" not in parsed:
        raise ValueError("missing field: legend_is_classed")
    lic = parsed["legend_is_classed"]
    if lic is not None and not isinstance(lic, bool):
        raise ValueError(
            f"legend_is_classed must be bool or null, got {type(lic).__name__}"
        )
    return parsed


def _process_step3_one(client, model, uid, image_path, step1_raw, step2_raw):
    last_error = None
    for attempt in range(mt_config.RETRY_ATTEMPTS):
        try:
            result = _call_step3(client, model, image_path, step1_raw, step2_raw)
            parsed = _validate_step3_response(result["parsed"])
            methods_count = sum(1 for f in _METHOD_FIELDS if parsed[f])
            return {
                "uid": uid,
                "status": "done",
                "model": model,
                "raw_response": result["raw"],
                "methods_count": methods_count,
                "confidence": parsed["confidence"],
                "legend_is_classed": parsed["legend_is_classed"],
                **{f: parsed[f] for f in _METHOD_FIELDS},
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "cache_read": result["cache_read"],
                "cache_creation": result["cache_creation"],
                "error": None,
            }
        except (json.JSONDecodeError, ValueError) as e:
            last_error = f"parse: {e}"
            if attempt < mt_config.RETRY_ATTEMPTS - 1:
                time.sleep(mt_config.RETRY_DELAY_S)
                continue
            return _step3_failure(uid, model, "parse_error", last_error)
        except Exception as e:
            last_error = str(e)
            txt = last_error.lower()
            if "rate_limit" in txt or "429" in txt or "overloaded" in txt or "529" in txt:
                wait = mt_config.RETRY_DELAY_S * (attempt + 1) * mt_config.RATE_LIMIT_BACKOFF_MULTIPLIER
                time.sleep(wait)
                continue
            if attempt < mt_config.RETRY_ATTEMPTS - 1:
                time.sleep(mt_config.RETRY_DELAY_S)
                continue
            return _step3_failure(uid, model, "error", last_error)
    return _step3_failure(uid, model, "error", last_error or "max retries exhausted")


def _step3_failure(uid, model, status, error):
    base = {f: None for f in _METHOD_FIELDS}
    return {
        "uid": uid, "status": status, "model": model, "raw_response": None,
        "methods_count": None, "confidence": None, "legend_is_classed": None,
        **base,
        "input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_creation": 0,
        "error": error,
    }


def run_step3(scope, force=False, model=None, workers=None, db_path=None):
    model = model or mt_config.MODEL_PER_STEP["step3"]
    workers = workers or mt_config.WORKERS

    print(f"\n[step3] model = {model}   workers = {workers}")
    print(f"[step3] gate (Python): is_stat=1 ∧ admin=1 ∧ has_quant=1 ∧ lang='en'")

    conn = mt_db.get_connection(db_path)
    rows = mt_db.select_candidates_for_step3(conn, scope, force=force)
    print(f"[step3] candidates to evaluate: {len(rows):,}")
    if not rows:
        conn.close()
        return {"evaluated": 0, "done": 0, "errors": 0}

    client = _get_client()
    done = errors = parse_errors = 0
    total_in = total_out = total_cr = total_cw = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for row in rows:
            image_path = row["local_path"]
            step1_raw = row["mt_step1_raw_response"]
            step2_raw = row["mt_step2_raw_response"]
            if not image_path or not Path(image_path).exists():
                mt_db.upsert_step3_result(conn, row["uid"], status="error", model=model,
                                            raw_response=None, error="local image file not found")
                errors += 1
                continue
            if not step1_raw or not step2_raw:
                mt_db.upsert_step3_result(conn, row["uid"], status="error", model=model,
                                            raw_response=None, error="missing step1/step2 raw response — rerun upstream")
                errors += 1
                continue
            futures[executor.submit(
                _process_step3_one, client, model, row["uid"], image_path, step1_raw, step2_raw
            )] = row["uid"]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="step3"):
            res = fut.result()
            mt_db.upsert_step3_result(
                conn, res["uid"],
                status=res["status"], model=res["model"], raw_response=res["raw_response"],
                has_choropleth=res["has_choropleth"], has_diagrams=res["has_diagrams"],
                has_dot_density=res["has_dot_density"], has_isolines=res["has_isolines"],
                has_cartogram=res["has_cartogram"], has_flow_map=res["has_flow_map"],
                has_heat_map=res["has_heat_map"],
                methods_count=res["methods_count"],
                legend_is_classed=res.get("legend_is_classed"),
                confidence=res["confidence"],
                input_tokens=res["input_tokens"], output_tokens=res["output_tokens"],
                cache_read=res["cache_read"], cache_creation=res["cache_creation"],
                error=res["error"],
            )
            conn.commit()
            total_in += res["input_tokens"]
            total_out += res["output_tokens"]
            total_cr += res["cache_read"]
            total_cw += res["cache_creation"]
            if res["status"] == "done":
                done += 1
            elif res["status"] == "parse_error":
                parse_errors += 1
            else:
                errors += 1

    conn.close()

    cost = _estimate_cost_haiku_or_sonnet(model, total_in, total_out, total_cr, total_cw)
    print(f"[step3] done: {done:,}   parse_errors: {parse_errors:,}   errors: {errors:,}")
    print(f"[step3] tokens — in: {total_in:,}  out: {total_out:,}  cache_read: {total_cr:,}  cache_write: {total_cw:,}")
    print(f"[step3] estimated cost: ${cost:.4f}")

    return {
        "evaluated": len(rows), "done": done, "parse_errors": parse_errors, "errors": errors,
        "tokens": {"input": total_in, "output": total_out, "cache_read": total_cr, "cache_creation": total_cw},
        "cost_usd": cost,
    }


def _estimate_cost_haiku_or_sonnet(model, in_tok, out_tok, cache_read, cache_write):
    if "haiku" in model.lower():
        rates = {"in": 1.00, "out": 5.00, "cr": 0.10, "cw": 1.25}
    elif "sonnet" in model.lower():
        rates = {"in": 3.00, "out": 15.00, "cr": 0.30, "cw": 3.75}
    else:
        rates = {"in": 3.00, "out": 15.00, "cr": 0.30, "cw": 3.75}
    return (
        in_tok * rates["in"] / 1_000_000
        + out_tok * rates["out"] / 1_000_000
        + cache_read * rates["cr"] / 1_000_000
        + cache_write * rates["cw"] / 1_000_000
    )


def run(steps, scope, force=False, model=None, workers=None, threshold=None, db_path=None):
    for s in steps:
        if s not in mt_config.IMPLEMENTED_STEPS:
            raise NotImplementedError(
                f"Step '{s}' is not implemented yet. "
                f"Implemented: {mt_config.IMPLEMENTED_STEPS}"
            )

    requested = set(steps) & {"size_gate", "step1", "step2", "step3", "step4"}
    if not requested:
        raise ValueError(
            "No runnable step requested. Pick from: size_gate, step1, step2, step3, step4."
        )

    stop_after = max(requested, key=mt_config.order_index)
    return run_phase1(
        scope,
        stop_after=stop_after,
        force=force,
        model=model,
        workers=workers,
        threshold=threshold,
        db_path=db_path,
    )


_STEP3_OTHER_METHODS = (
    "has_diagrams", "has_dot_density",
    "has_isolines", "has_cartogram", "has_flow_map", "has_heat_map",
)


def _apply_step4_formal(methods, legend_is_classed):
    has_choro = methods.get("has_choropleth") == 1
    other_present = [m for m in _STEP3_OTHER_METHODS if methods.get(m) == 1]
    f2 = has_choro and not other_present
    f3 = legend_is_classed == 1
    passed = f2 and f3
    return {
        "f2_passed": bool(f2),
        "f3_passed": bool(f3),
        "passed": bool(passed),
        "other_methods_present": other_present,
    }


def run_step4(scope, force=False, db_path=None):
    print(f"\n[step4] formal filter — F2 (choropleth-only) ∧ F3 (legend classed)")

    conn = mt_db.get_connection(db_path)
    method_cols = ["mt_has_choropleth"] + [f"mt_{m}" for m in _STEP3_OTHER_METHODS]
    method_cols.append("mt_legend_is_classed")
    base = (f"SELECT uid, {', '.join(method_cols)} "
            f"FROM candidates WHERE mt_step3_status='done'")
    where, params = mt_db._scope_where(scope)
    sql = base + where
    if not force:
        sql += " AND mt_step4_status IS NULL"
    if scope.get("pilot"):
        sql += " ORDER BY uid LIMIT ?"
        params.append(int(scope["pilot"]))
    rows = conn.execute(sql, params).fetchall()
    print(f"[step4] candidates to evaluate: {len(rows):,}")
    if not rows:
        conn.close()
        return {"evaluated": 0, "passed": 0, "failed": 0}

    n_pass = n_fail = 0
    for row in tqdm(rows, desc="step4"):
        methods = {"has_choropleth": row["mt_has_choropleth"]}
        for m in _STEP3_OTHER_METHODS:
            methods[m] = row[f"mt_{m}"]
        verdict = _apply_step4_formal(methods, row["mt_legend_is_classed"])
        mt_db.upsert_step4_result(
            conn, row["uid"], status="done",
            passed=verdict["passed"],
            f2_passed=verdict["f2_passed"],
            f3_passed=verdict["f3_passed"],
        )
        if verdict["passed"]:
            n_pass += 1
        else:
            n_fail += 1
    conn.commit()
    conn.close()

    print(f"[step4] passed: {n_pass:,}   failed: {n_fail:,}   (of {len(rows):,} step3-done)")
    return {"evaluated": len(rows), "passed": n_pass, "failed": n_fail}


def _load_uid_state(conn, uid):
    other_method_cols = ", ".join(f"mt_{m}" for m in _STEP3_OTHER_METHODS)
    row = conn.execute(
        "SELECT mt_size_status, mt_step1_status, mt_step1_raw_response, "
        "       mt_is_map, mt_is_map_dominant, "
        "       mt_step2_status, mt_step2_raw_response, "
        "       mt_is_statistical_map, mt_has_admin_units, "
        "       mt_has_quantitative_data, mt_map_language, mt_has_text, "
        "       mt_step3_status, mt_has_choropleth, "
        f"      {other_method_cols}, "
        "       mt_legend_is_classed, "
        "       mt_step4_status, image_max_dim, image_width, image_height "
        "FROM candidates WHERE uid = ?",
        (uid,),
    ).fetchone()
    return dict(row) if row else {}


def _gate1_pass_from_step1(res):
    return res.get("status") == "done" and bool(res.get("is_map")) and bool(res.get("is_map_dominant"))


def _gate1_pass_from_state(state):
    return state.get("mt_is_map") == 1 and state.get("mt_is_map_dominant") == 1


def _gate2_pass_from_step2(res):
    if res.get("status") != "done":
        return False
    if not (res.get("is_statistical_map") and res.get("has_admin_units") and res.get("has_quantitative_data")):
        return False
    return res.get("map_language") == "en"


def _gate2_pass_from_state(state):
    if not (state.get("mt_is_statistical_map") == 1
            and state.get("mt_has_admin_units") == 1
            and state.get("mt_has_quantitative_data") == 1):
        return False
    return state.get("mt_map_language") == "en"


def _accumulate_tokens(totals, res):
    for k_src, k_dst in (("input_tokens", "input"),
                         ("output_tokens", "output"),
                         ("cache_read", "cache_read"),
                         ("cache_creation", "cache_creation")):
        totals[k_dst] += int(res.get(k_src) or 0)


def _chain_phase1_one(client, uid, image_path, image_w, image_h,
                       models, force, db_path, threshold, stop_after):
    STEP_ORDER = {"size_gate": 0, "step1": 1, "step2": 2, "step3": 3, "step4": 4}
    stop_idx = STEP_ORDER[stop_after]

    outcome = {
        "uid": uid,
        "size_gate": "skipped",
        "step1": "skipped",
        "step2": "skipped",
        "step3": "skipped",
        "step4": "skipped",
        "tokens": {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0},
    }

    conn = mt_db.get_connection(db_path)
    try:
        state = _load_uid_state(conn, uid)

        if force or not state.get("mt_size_status"):
            if image_w is None or image_h is None:
                sg_status, min_dim = "fail_too_small", 0
            else:
                min_dim = min(image_w, image_h)
                sg_status = "pass" if min_dim >= threshold else "fail_too_small"
            mt_db.upsert_size_gate(conn, uid, sg_status, min_dim)
            conn.commit()
            outcome["size_gate"] = sg_status
        else:
            outcome["size_gate"] = state["mt_size_status"] + "_cached"

        if not outcome["size_gate"].startswith("pass"):
            return outcome
        if stop_idx < STEP_ORDER["step1"]:
            return outcome

        step1_needs_run = force or state.get("mt_step1_status") not in ("done",)

        if step1_needs_run:
            if not image_path or not Path(image_path).exists():
                outcome["step1"] = "skipped_no_file"
                return outcome
            r1 = _process_step1_one(client, models["step1"], uid, image_path)
            mt_db.upsert_step1_result(conn, uid,
                status=r1["status"], model=r1["model"], raw_response=r1["raw_response"],
                is_map=r1["is_map"], is_map_dominant=r1["is_map_dominant"],
                short_description=r1["short_description"],
                input_tokens=r1["input_tokens"], output_tokens=r1["output_tokens"],
                cache_read=r1["cache_read"], cache_creation=r1["cache_creation"],
                error=r1["error"])
            conn.commit()
            _accumulate_tokens(outcome["tokens"], r1)
            outcome["step1"] = r1["status"]
            step1_raw = r1["raw_response"]
            gate1_ok = _gate1_pass_from_step1(r1)
        else:
            outcome["step1"] = "done_cached"
            step1_raw = state.get("mt_step1_raw_response")
            gate1_ok = _gate1_pass_from_state(state)

        if not gate1_ok or not step1_raw:
            return outcome
        if stop_idx < STEP_ORDER["step2"]:
            return outcome

        step2_needs_run = force or state.get("mt_step2_status") not in ("done",)
        if step2_needs_run:
            if not image_path or not Path(image_path).exists():
                outcome["step2"] = "skipped_no_file"
                return outcome
            r2 = _process_step2_one(client, models["step2"], uid, image_path, step1_raw)
            mt_db.upsert_step2_result(conn, uid,
                status=r2["status"], model=r2["model"], raw_response=r2["raw_response"],
                map_language=r2["map_language"],
                is_statistical_map=r2["is_statistical_map"],
                has_admin_units=r2["has_admin_units"],
                data_level=r2["data_level"],
                has_text=r2["has_text"],
                has_quantitative_data=r2["has_quantitative_data"],
                input_tokens=r2["input_tokens"], output_tokens=r2["output_tokens"],
                cache_read=r2["cache_read"], cache_creation=r2["cache_creation"],
                error=r2["error"])
            conn.commit()
            _accumulate_tokens(outcome["tokens"], r2)
            outcome["step2"] = r2["status"]
            step2_raw = r2["raw_response"]
            gate2_ok = _gate2_pass_from_step2(r2)
        else:
            outcome["step2"] = "done_cached"
            step2_raw = state.get("mt_step2_raw_response")
            gate2_ok = _gate2_pass_from_state(state)

        if not gate2_ok or not step2_raw:
            return outcome
        if stop_idx < STEP_ORDER["step3"]:
            return outcome

        step3_needs_run = force or state.get("mt_step3_status") not in ("done",)
        if step3_needs_run:
            if not image_path or not Path(image_path).exists():
                outcome["step3"] = "skipped_no_file"
                return outcome
            r3 = _process_step3_one(client, models["step3"], uid, image_path, step1_raw, step2_raw)
            mt_db.upsert_step3_result(conn, uid,
                status=r3["status"], model=r3["model"], raw_response=r3["raw_response"],
                has_choropleth=r3["has_choropleth"], has_diagrams=r3["has_diagrams"],
                has_dot_density=r3["has_dot_density"], has_isolines=r3["has_isolines"],
                has_cartogram=r3["has_cartogram"], has_flow_map=r3["has_flow_map"],
                has_heat_map=r3["has_heat_map"],
                methods_count=r3["methods_count"],
                legend_is_classed=r3.get("legend_is_classed"),
                confidence=r3["confidence"],
                input_tokens=r3["input_tokens"], output_tokens=r3["output_tokens"],
                cache_read=r3["cache_read"], cache_creation=r3["cache_creation"],
                error=r3["error"])
            conn.commit()
            _accumulate_tokens(outcome["tokens"], r3)
            outcome["step3"] = r3["status"]
            state = _load_uid_state(conn, uid)
        else:
            outcome["step3"] = "done_cached"

        if state.get("mt_step3_status") != "done":
            return outcome
        if stop_idx < STEP_ORDER["step4"]:
            return outcome

        step4_needs_run = force or state.get("mt_step4_status") != "done"
        if step4_needs_run:
            methods = {"has_choropleth": state.get("mt_has_choropleth")}
            for m in _STEP3_OTHER_METHODS:
                methods[m] = state.get(f"mt_{m}")
            verdict = _apply_step4_formal(methods, state.get("mt_legend_is_classed"))
            mt_db.upsert_step4_result(conn, uid,
                status="done", passed=verdict["passed"],
                f2_passed=verdict["f2_passed"],
                f3_passed=verdict["f3_passed"])
            conn.commit()
            outcome["step4"] = "passed" if verdict["passed"] else "failed"
        else:
            outcome["step4"] = "done_cached"

        return outcome
    finally:
        conn.close()


def _select_chain_scope_phase1(conn, scope):
    base = ("SELECT uid, local_path, image_width, image_height "
            "FROM candidates "
            "WHERE download_status='success' ")
    where, params = mt_db._scope_where(scope)
    sql = base + where
    if scope.get("pilot"):
        sql += " ORDER BY uid LIMIT ?"
        params.append(int(scope["pilot"]))
    return conn.execute(sql, params).fetchall()


def run_phase1(scope, stop_after="step4", force=False, model=None,
                workers=None, threshold=None, db_path=None):
    if stop_after not in ("size_gate", "step1", "step2", "step3", "step4"):
        raise ValueError(f"stop_after must be one of size_gate/step1/step2/step3/step4, got {stop_after!r}")

    added = mt_db.migrate_schema_mt(db_path)
    if added:
        print(f"[init] schema migrated — added columns: {added}")

    if not any(scope.get(k) for k in ("all", "pilot", "uids")):
        raise ValueError("scope must specify exactly one of: all=True, pilot=N, uids=[...]")

    workers = workers or mt_config.WORKERS
    threshold = threshold or base_config.MIN_IMAGE_DIM_PX
    models = {
        "step1": model or mt_config.MODEL_PER_STEP["step1"],
        "step2": model or mt_config.MODEL_PER_STEP["step2"],
        "step3": model or mt_config.MODEL_PER_STEP["step3"],
    }

    print("=" * 60)
    print(f"PHASE 1 CHAIN  ({mt_config.PIPELINE_VERSION})")
    print(f"Scope: {_scope_describe(scope)}")
    print(f"Stop after: {stop_after}   Workers: {workers}   Size threshold: {threshold}px (baseline)")
    print(f"Models: step1={models['step1']}  step2={models['step2']}  step3={models['step3']}")
    print(f"Step4 formal filter: F2 choropleth-only ∧ F3 legend classed  (Python only, no VLM)")
    print(f"Force re-run: {force}")
    print("=" * 60)

    conn = mt_db.get_connection(db_path)
    rows = _select_chain_scope_phase1(conn, scope)
    conn.close()
    print(f"[phase1] uids to consider: {len(rows):,}")
    if not rows:
        return {"considered": 0}

    client = _get_client()

    counters = {
        "considered": len(rows),
        "size_gate": {"pass": 0, "fail_too_small": 0, "cached": 0, "skipped": 0},
        "step1": {"done": 0, "cached": 0, "error": 0, "parse_error": 0, "skipped": 0, "no_file": 0},
        "step2": {"done": 0, "cached": 0, "error": 0, "parse_error": 0, "skipped": 0, "no_file": 0},
        "step3": {"done": 0, "cached": 0, "error": 0, "parse_error": 0, "skipped": 0, "no_file": 0},
        "step4": {"passed": 0, "failed": 0, "cached": 0, "skipped": 0},
        "tokens": {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0},
    }

    def _bump(step_key, value):
        if value == "skipped":
            counters[step_key]["skipped"] += 1
        elif value == "skipped_no_file":
            counters[step_key]["no_file"] += 1
        elif value.endswith("_cached"):
            counters[step_key]["cached"] += 1
        elif value in ("pass", "fail_too_small", "passed", "failed"):
            counters[step_key][value] += 1
        elif value in ("done", "error", "parse_error"):
            counters[step_key][value] += 1
        else:
            counters[step_key].setdefault("other", 0)
            counters[step_key]["other"] += 1

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for row in rows:
            futures[executor.submit(
                _chain_phase1_one, client, row["uid"], row["local_path"],
                row["image_width"], row["image_height"],
                models, force, db_path, threshold, stop_after,
            )] = row["uid"]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="phase1"):
            res = fut.result()
            _bump("size_gate", res["size_gate"])
            _bump("step1", res["step1"])
            _bump("step2", res["step2"])
            _bump("step3", res["step3"])
            _bump("step4", res["step4"])
            for k in counters["tokens"]:
                counters["tokens"][k] += res["tokens"][k]

    haiku_in = counters["tokens"]["input"]
    est_cost = _estimate_cost_haiku_or_sonnet(
        models["step1"],
        counters["tokens"]["input"], counters["tokens"]["output"],
        counters["tokens"]["cache_read"], counters["tokens"]["cache_creation"],
    )

    print("\n" + "=" * 60)
    print("PHASE 1 DONE")
    for step in ("size_gate", "step1", "step2", "step3", "step4"):
        print(f"  [{step}] {counters[step]}")
    print(f"  [tokens] {counters['tokens']}")
    print(f"  [estimated cost, single-model rough] ~${est_cost:.4f}")
    print("=" * 60)
    return counters


def _scope_describe(scope):
    if scope.get("all"):
        return "ALL downloaded candidates"
    if scope.get("pilot"):
        return f"pilot N={scope['pilot']}"
    if scope.get("uids"):
        uids = list(scope["uids"])
        if len(uids) <= 5:
            return f"uids={uids}"
        return f"uids=[{len(uids)} total: {uids[:3]} ...]"
    return "(unspecified)"
