"""校验 Phase 7 偏好 JSONL，不加载模型或启动训练。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from med_align_rag_4b.dpo.config import load_dpo_config
from med_align_rag_4b.dpo.data import (
    load_blocked_fingerprints,
    load_preference_dataset,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--blocked-fingerprints",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--require-reviewed", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_dpo_config(args.config)
    blocked = load_blocked_fingerprints(args.blocked_fingerprints)
    summary = load_preference_dataset(
        args.data,
        config,
        blocked_fingerprints=blocked,
        require_reviewed=args.require_reviewed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
