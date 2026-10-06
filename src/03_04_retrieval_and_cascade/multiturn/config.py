from pathlib import Path

import config as base_config

PROMPTS_DIR = Path(__file__).parent / "prompts"

PIPELINE_VERSION = "mt_v1"

MODEL_PER_STEP = {
    "step1":  base_config.CLAUDE_MODEL_HAIKU,
    "step2":  base_config.CLAUDE_MODEL_HAIKU,
    "step3":  base_config.CLAUDE_MODEL_SONNET,
    "step4":  None,
}

MAX_TOKENS_PER_STEP = {
    "step1":  400,
    "step2":  600,
    "step3":  700,
    "step4":  0,
}

WORKERS = 5

RETRY_ATTEMPTS = 3
RETRY_DELAY_S = 5
RATE_LIMIT_BACKOFF_MULTIPLIER = 1

STEP_ORDER = ["size_gate", "step1", "step2", "step3", "step4"]

IMPLEMENTED_STEPS = ["size_gate", "step1", "step2", "step3", "step4"]


def is_implemented(step: str) -> bool:
    return step in IMPLEMENTED_STEPS


def order_index(step: str) -> int:
    return STEP_ORDER.index(step)


def steps_up_to(step: str) -> list:
    idx = order_index(step)
    return STEP_ORDER[: idx + 1]
