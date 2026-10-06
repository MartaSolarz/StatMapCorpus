#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
import db
import pipeline


def cmd_sample(args):
    print("=" * 60)
    print("Sampling the candidates")
    print("=" * 60)

    exclude = args.exclude_domains.split(",") if args.exclude_domains else None

    db.init_db()
    inserted, skipped = pipeline.smart_sample(
        pred_proba_min=args.pred_proba_min,
        domain_cap=args.domain_cap,
        sample_target=args.sample_target,
        exclude_domains=exclude,
        seed=args.seed,
    )

    print(f"\nDone. Run 'validate-urls' next.")


def cmd_validate_urls(args):
    print("=" * 60)
    print("STEP 2: URL Validation")
    print("=" * 60)

    db.init_db()
    pipeline.validate_urls()

    print(f"\nDone. Run 'download' next.")


def cmd_download(args):
    print("=" * 60)
    print("STEP 3: Image Download + F1 Check")
    print("=" * 60)

    db.init_db()
    pipeline.download_images()

    print(f"\nDone. Run 'ai-check' next.")


def cmd_status(args):
    db.init_db()
    conn = db.get_connection()
    stats = db.get_stats(conn)
    conn.close()

    print("=" * 60)
    print("PIPELINE STATUS")
    print("=" * 60)

    total = stats["total"]
    if total == 0:
        print("\n  No candidates yet. Run 'sample' first.")
        return

    print(f"\n  Candidates:         {total:>8,}")
    print(f"  ─────────────────────────────")

    url_done = stats["url_alive"] + stats["url_dead"]
    print(f"  URL alive:          {stats['url_alive']:>8,}  "
          f"({stats['url_alive']/total*100:.1f}%)" if total else "")
    print(f"  URL dead/error:     {stats['url_dead']:>8,}")
    print(f"  URL pending:        {stats['url_pending']:>8,}")
    print(f"  ─────────────────────────────")

    if stats['downloaded']:
        print(f"  Downloaded:         {stats['downloaded']:>8,}")
    print(f"  Download failed:    {stats['download_failed']:>8,}")
    print(f"  Download pending:   {stats['download_pending']:>8,}")
    print(f"  ─────────────────────────────")

    res = stats.get("resolution")
    if res and res["count"] and res["count"] > 0:
        print(f"  Resolution (downloaded images):")
        print(f"    Range W:  {res['min_w']}–{res['max_w']} px  (avg {res['avg_w']:.0f})")
        print(f"    Range H:  {res['min_h']}–{res['max_h']} px  (avg {res['avg_h']:.0f})")
        print(f"    < 400px:  {res['below_400']:,}")
        print(f"    < 600px:  {res['below_600']:,}")
        print(f"    < 800px:  {res['below_800']:,}")
        print(f"  ─────────────────────────────")

    if "mt_size_pass" in stats:
        print(f"  Size gate pass:     {stats['mt_size_pass']:>8,}")
        print(f"  Size gate fail:     {stats['mt_size_fail']:>8,}  (< {config.MIN_IMAGE_DIM_PX}px)")
        print(f"  ─────────────────────────────")

    for step in ("step1", "step2", "step3"):
        done_key = f"mt_{step}_done"
        err_key = f"mt_{step}_error"
        if done_key in stats:
            print(f"  {step} done:           {stats[done_key]:>8,}")
            if stats.get(err_key):
                print(f"  {step} error:          {stats[err_key]:>8,}")

    if "passes_dataset" in stats:
        print(f"  ─────────────────────────────")
        print(f"  Corpus (passed Gate 4): {stats['passes_dataset']:>8,}")
    if "passes_subset" in stats:
        print(f"  step4 flag (separate project): {stats['passes_subset']:>8,}")

    m = stats.get("methods")
    if m and any(m[k] for k in m.keys() if m[k]):
        print(f"\n  Methods (among step3-done):")
        for label, key in [
            ("Choropleth", "choropleth"),
            ("Diagrams",   "diagrams"),
            ("Isolines",   "isolines"),
            ("Dot density","dot_density"),
            ("Heat map",   "heat_map"),
            ("Cartogram",  "cartogram"),
            ("Flow map",   "flow_map"),
        ]:
            print(f"    {label:14}    {m[key] or 0:,}")

    if stats["batches"]:
        print(f"\n  Batches:")
        for b in stats["batches"]:
            params = b["params"] or ""
            print(f"    #{b['id']} {b['step']}: {b['candidates_count']:,} candidates "
                  f"({b['started_at'][:10] if b['started_at'] else '?'})")


def _resolve_scope(args):
    scope_options = sum(1 for x in (args.all, args.pilot, args.uids) if x)
    if scope_options != 1:
        print("ERROR: pick exactly ONE of --all, --pilot N, --uids a,b,c", file=sys.stderr)
        sys.exit(2)
    if args.all:
        return {"all": True}
    if args.pilot:
        return {"pilot": args.pilot}
    return {"uids": [u.strip() for u in args.uids.split(",") if u.strip()]}


def cmd_multiturn(args):
    from multiturn import pipeline as mt_pipeline
    from multiturn import config as mt_config

    if args.through:
        steps = mt_config.steps_up_to(args.through)
        steps = [s for s in steps if s in mt_config.IMPLEMENTED_STEPS]
    elif args.steps:
        steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    else:
        steps = list(mt_config.IMPLEMENTED_STEPS)

    scope = _resolve_scope(args)

    summary = mt_pipeline.run(
        steps=steps,
        scope=scope,
        force=args.force,
        model=args.model,
        workers=args.workers,
        threshold=args.size_threshold,
    )
    print("\nSummary:")
    for step, info in summary.items():
        print(f"  [{step}] {info}")


def cmd_phase1(args):
    from multiturn import pipeline as mt_pipeline

    scope = _resolve_scope(args)
    summary = mt_pipeline.run_phase1(
        scope=scope,
        stop_after=args.stop_after,
        force=args.force,
        model=args.model,
        workers=args.workers,
        threshold=args.size_threshold,
    )
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  [{k}] {v}")


def main():
    parser = argparse.ArgumentParser(
        description="StatMapCorpus Stages 3-4: retrieval and the vision-language cascade",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline step")

    p_sample = subparsers.add_parser("sample",
        help="Draw the candidates: pred_proba above the threshold, at most DOMAIN_CAP per domain")
    p_sample.add_argument("--pred-proba-min", type=float, default=config.PRED_PROBA_MIN,
                          help=f"Min pred_proba (default: {config.PRED_PROBA_MIN})")
    p_sample.add_argument("--domain-cap", type=int, default=config.DOMAIN_CAP,
                          help=f"Max per domain (default: {config.DOMAIN_CAP})")
    p_sample.add_argument("--sample-target", type=int, default=config.SAMPLE_TARGET,
                          help=f"Target sample size (default: {config.SAMPLE_TARGET})")
    p_sample.add_argument("--exclude-domains", type=str, default=None,
                          help="Comma-separated domains to exclude")
    p_sample.add_argument("--seed", type=int, default=None,
                          help="Random seed (default: current timestamp = new sample each run)")
    p_sample.set_defaults(func=cmd_sample)

    p_urls = subparsers.add_parser("validate-urls", help="Check URL availability")
    p_urls.set_defaults(func=cmd_validate_urls)

    p_dl = subparsers.add_parser("download", help="Download images + F1 check")
    p_dl.set_defaults(func=cmd_download)

    p_status = subparsers.add_parser("status", help="Show pipeline progress")
    p_status.set_defaults(func=cmd_status)

    p_p1 = subparsers.add_parser("phase1",
        help="Phase 1 chain — per-image cascade: size_gate + step1 + step2 + step3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Full cascade on 100-map pilot:\n"
            "    python3 run_pipeline.py phase1 --pilot 100\n\n"
            "  Full scan (all downloaded), stop after step2:\n"
            "    python3 run_pipeline.py phase1 --all --stop-after step2\n\n"
            "  Rerun cascade for specific uids (force overrides prior results):\n"
            "    python3 run_pipeline.py phase1 --uids abc,def --force\n"
        ))
    p_p1.add_argument("--all", action="store_true",
        help="Chain over ALL downloaded candidates.")
    p_p1.add_argument("--pilot", type=int, default=None,
        help="Pilot mode — chain over the first N eligible candidates.")
    p_p1.add_argument("--uids", type=str, default=None,
        help="Comma-separated UIDs to process.")
    p_p1.add_argument("--stop-after", dest="stop_after", type=str, default="step4",
        choices=["size_gate", "step1", "step2", "step3", "step4"],
        help="Chain up to (and including) this step. Default: step4 (includes formal filter).")
    p_p1.add_argument("--force", action="store_true",
        help="Re-run each step even if its result is already in the DB.")
    p_p1.add_argument("--model", type=str, default=None,
        help="Override model for ALL VLM steps (default: per-step config).")
    p_p1.add_argument("--workers", type=int, default=None,
        help="Concurrent uids in flight (default: multiturn config).")
    p_p1.add_argument("--size-threshold", dest="size_threshold", type=int, default=None,
        help=f"Override size_gate min-dim (default: {config.MIN_IMAGE_DIM_PX}).")
    p_p1.set_defaults(func=cmd_phase1)

    p_mt = subparsers.add_parser("multiturn",
        help="[LEGACY] Multiturn pipeline — routes to phase1 based on --steps/--through",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Run size_gate + step1 on a 100-map pilot:\n"
            "    python3 run_pipeline.py multiturn --through step1 --pilot 100\n\n"
            "  Run ONLY size_gate on every downloaded candidate:\n"
            "    python3 run_pipeline.py multiturn --steps size_gate --all\n\n"
            "  Run step1 (size_gate must already be done) on specific UIDs:\n"
            "    python3 run_pipeline.py multiturn --steps step1 --uids abc...,def...\n\n"
            "  Re-run step1 for a few UIDs (overwrites prior results):\n"
            "    python3 run_pipeline.py multiturn --steps step1 --uids xyz... --force\n"
        ))
    p_mt.add_argument("--steps", type=str, default=None,
        help="Comma-separated step names to run (e.g. 'size_gate,step1'). "
             "Default: all implemented steps in order.")
    p_mt.add_argument("--through", type=str, default=None,
        help="Run all implemented steps up to and including this step "
             "(e.g. --through step1).")
    p_mt.add_argument("--all", action="store_true",
        help="Run on ALL downloaded candidates (full scan).")
    p_mt.add_argument("--pilot", type=int, default=None,
        help="Pilot mode — run on the first N eligible candidates.")
    p_mt.add_argument("--uids", type=str, default=None,
        help="Comma-separated UIDs to process.")
    p_mt.add_argument("--force", action="store_true",
        help="Re-run a step even if its result is already in the DB (overwrites).")
    p_mt.add_argument("--model", type=str, default=None,
        help="Override VLM model for VLM steps (default: per-step config).")
    p_mt.add_argument("--workers", type=int, default=None,
        help=f"Concurrent VLM workers (default: multiturn config).")
    p_mt.add_argument("--size-threshold", type=int, default=None,
        help=f"Override size_gate min-dim threshold (default: {config.MIN_IMAGE_DIM_PX}px).")
    p_mt.set_defaults(func=cmd_multiturn)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
