#!/usr/bin/env python3
"""
Build Level-2 manifests for L2-specific Skill Evolution v1
==========================================================

Research goal
-------------
After the frozen L1 Skill failed to transfer to L2, create a CLEAN L2-only
development/evaluation protocol to test whether Skill Evolution must adapt
to task difficulty.

This builder creates:
  - L2 dev Fold A: 4 four_letter_tf + 4 scalar_range
  - L2 dev Fold B: 4 four_letter_tf + 4 scalar_range
  - L2 frozen holdout: 20 four_letter_tf + 30 scalar_range

Total:
  - development = 16 cases
  - holdout     = 50 cases

Important exclusions
--------------------
The builder excludes every case already used by:
  - prior L1 Fold A
  - prior L1 Fold B
  - prior L1 Holdout
  - L1 External Test v1
  - L1->L2 Cross-Level Test v1

Selection uses metadata diversity only.
NO model outputs, GT scores, or failure labels are used.

The files are frozen after creation and cannot be overwritten silently.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(
    "/home/training/automatic_prompt_engineer/"
    "hermes_factorybench_skill_experiment"
)
FB_ROOT = Path(
    "/home/training/automatic_prompt_engineer/FactoryBench"
)

MANIFEST_DIR = ROOT / "data_manifests/meta_m1"

OUT_A = MANIFEST_DIR / "factorybench_l2_skill_dev_fold_a.json"
OUT_B = MANIFEST_DIR / "factorybench_l2_skill_dev_fold_b.json"
OUT_H = MANIFEST_DIR / "factorybench_l2_skill_holdout.json"
OUT_LOCK = MANIFEST_DIR / "factorybench_l2_skill_v1_lock.json"

EXCLUDE_MANIFESTS = [
    MANIFEST_DIR / "factorybench_l123_dev_fold_a.json",
    MANIFEST_DIR / "factorybench_l123_dev_fold_b.json",
    MANIFEST_DIR / "factorybench_l123_holdout.json",
    MANIFEST_DIR / "factorybench_l1_external_test_v1.json",
    MANIFEST_DIR / "factorybench_l2_cross_level_test_v1.json",
]

DEV_PER_FORMAT = {
    "four_letter_tf": 8,
    "scalar_range": 8,
}

HOLDOUT_PER_FORMAT = {
    "four_letter_tf": 20,
    "scalar_range": 30,
}

SELECTION_SEED = "factorybench-l2-skill-evolution-v1-20260821"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(
            f"Refusing to overwrite frozen manifest: {path}"
        )
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(value: Any) -> str:
    return "unknown" if value is None else str(value)


def main() -> None:
    for p in [OUT_A, OUT_B, OUT_H, OUT_LOCK]:
        if p.exists():
            raise RuntimeError(
                "L2 Skill Evolution v1 manifests already exist. "
                f"Do not overwrite: {p}"
            )

    sys.path.insert(0, str(FB_ROOT))
    from factorybench import load_split

    sys.path.insert(0, str(ROOT))
    from scripts.meta_coevo import run_static_surrogate as base

    excluded_ids: set[str] = set()
    exclusion_report = {}

    for path in EXCLUDE_MANIFESTS:
        if not path.exists():
            raise FileNotFoundError(
                f"Required exclusion manifest missing: {path}"
            )

        payload = read_json(path)
        ids = [row["id"] for row in payload["items"]]
        excluded_ids.update(ids)
        exclusion_report[path.name] = len(ids)

    items = load_split(
        2,
        split="test",
        revision=base.REV,
        max_items=None,
    )

    available = [
        x for x in items
        if x.id not in excluded_ids
    ]

    format_counts = Counter(
        x.answer_format.value for x in available
    )

    print("========================================")
    print("L2 SKILL EVOLUTION V1 MANIFEST BUILDER")
    print("========================================")
    print("Dataset revision:", base.REV)
    print("L2 test total:", len(items))
    print("Excluded unique IDs:", len(excluded_ids))
    print("Available L2 after exclusions:", len(available))

    print("\nAvailable formats:")
    for fmt, n in sorted(format_counts.items()):
        print(f"  {fmt}: {n}")

    required = {
        fmt: DEV_PER_FORMAT.get(fmt, 0) + HOLDOUT_PER_FORMAT.get(fmt, 0)
        for fmt in set(DEV_PER_FORMAT) | set(HOLDOUT_PER_FORMAT)
    }

    for fmt, need in required.items():
        have = format_counts.get(fmt, 0)
        if have < need:
            raise RuntimeError(
                f"Not enough {fmt}: need={need}, have={have}"
            )

    def dataset_of(x):
        return safe(getattr(x, "dataset", None))

    def template_of(x):
        return getattr(x, "template_id", None)

    def episode_of(x):
        return (getattr(x, "provenance", None) or {}).get("episode")

    def stable_hash(x, scope: str):
        raw = f"{SELECTION_SEED}|{scope}|{x.id}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def choose_diverse(pool, n, scope):
        selected = []
        remaining = list(pool)

        seen_dataset = set()
        seen_template = set()
        seen_episode = set()

        while remaining and len(selected) < n:
            ranked = []

            for x in remaining:
                ds = dataset_of(x)
                tp = template_of(x)
                ep = episode_of(x)

                novelty = (
                    int(ds not in seen_dataset),
                    int(tp not in seen_template),
                    int(ep not in seen_episode),
                )

                ranked.append(
                    (
                        -novelty[0],
                        -novelty[1],
                        -novelty[2],
                        stable_hash(x, scope),
                        x,
                    )
                )

            ranked.sort(key=lambda row: row[:4])
            x = ranked[0][4]

            selected.append(x)
            seen_dataset.add(dataset_of(x))
            seen_template.add(template_of(x))
            seen_episode.add(episode_of(x))

            remaining = [y for y in remaining if y.id != x.id]

        if len(selected) != n:
            raise RuntimeError(
                f"{scope}: requested={n}, selected={len(selected)}"
            )

        return selected

    # --------------------------------------------------------------
    # First select the entire dev pool per format.
    # Then deterministically split it into A/B while keeping formats balanced.
    # --------------------------------------------------------------
    dev_by_format = {}
    remaining_after_dev = list(available)

    for fmt, n in DEV_PER_FORMAT.items():
        pool = [
            x for x in remaining_after_dev
            if x.answer_format.value == fmt
        ]

        chosen = choose_diverse(
            pool,
            n,
            scope=f"dev_{fmt}",
        )

        dev_by_format[fmt] = chosen
        chosen_ids = {x.id for x in chosen}
        remaining_after_dev = [
            x for x in remaining_after_dev
            if x.id not in chosen_ids
        ]

    fold_a_items = []
    fold_b_items = []

    for fmt, chosen in dev_by_format.items():
        ordered = sorted(
            chosen,
            key=lambda x: stable_hash(x, f"foldsplit_{fmt}")
        )

        half = len(ordered) // 2
        fold_a_items.extend(ordered[:half])
        fold_b_items.extend(ordered[half:])

    # --------------------------------------------------------------
    # Holdout is selected only AFTER development items are removed.
    # --------------------------------------------------------------
    holdout_items = []

    for fmt, n in HOLDOUT_PER_FORMAT.items():
        pool = [
            x for x in remaining_after_dev
            if x.answer_format.value == fmt
        ]

        chosen = choose_diverse(
            pool,
            n,
            scope=f"holdout_{fmt}",
        )

        holdout_items.extend(chosen)

        chosen_ids = {x.id for x in chosen}
        remaining_after_dev = [
            x for x in remaining_after_dev
            if x.id not in chosen_ids
        ]

    def row(x):
        return {
            "id": x.id,
            "level": 2,
            "split": "test",
            "episode": episode_of(x),
            "answer_format": x.answer_format.value,
            "dataset": getattr(x, "dataset", None),
            "template_id": template_of(x),
        }

    rows_a = [row(x) for x in fold_a_items]
    rows_b = [row(x) for x in fold_b_items]
    rows_h = [row(x) for x in holdout_items]

    ids_a = {x["id"] for x in rows_a}
    ids_b = {x["id"] for x in rows_b}
    ids_h = {x["id"] for x in rows_h}

    assert not (ids_a & ids_b)
    assert not (ids_a & ids_h)
    assert not (ids_b & ids_h)
    assert not ((ids_a | ids_b | ids_h) & excluded_ids)

    def manifest(name, purpose, rows):
        return {
            "name": name,
            "purpose": purpose,
            "dataset_revision": base.REV,
            "frozen_after_creation": True,
            "selection_method": (
                "deterministic metadata-diversity selection; "
                "no model outputs or GT performance used"
            ),
            "items": rows,
        }

    write_json_new(
        OUT_A,
        manifest(
            "factorybench_l2_skill_dev_fold_a",
            "Level-2 Skill Evolution v1 development Fold A",
            rows_a,
        ),
    )

    write_json_new(
        OUT_B,
        manifest(
            "factorybench_l2_skill_dev_fold_b",
            "Level-2 Skill Evolution v1 development Fold B",
            rows_b,
        ),
    )

    write_json_new(
        OUT_H,
        manifest(
            "factorybench_l2_skill_holdout",
            "Frozen Level-2 holdout for L2-specific Skill Evolution v1",
            rows_h,
        ),
    )

    lock = {
        "experiment": "l2_skill_evolution_v1",
        "dataset_revision": base.REV,
        "fold_a_path": str(OUT_A),
        "fold_b_path": str(OUT_B),
        "holdout_path": str(OUT_H),
        "fold_a_sha256": sha256(OUT_A),
        "fold_b_sha256": sha256(OUT_B),
        "holdout_sha256": sha256(OUT_H),
        "development_case_count": len(rows_a) + len(rows_b),
        "holdout_case_count": len(rows_h),
        "development_format_counts": dict(
            sorted(
                Counter(
                    x["answer_format"]
                    for x in rows_a + rows_b
                ).items()
            )
        ),
        "holdout_format_counts": dict(
            sorted(
                Counter(
                    x["answer_format"]
                    for x in rows_h
                ).items()
            )
        ),
        "excluded_prior_manifests": exclusion_report,
        "holdout_feedback_allowed": False,
        "holdout_rule_generation_allowed": False,
        "holdout_rule_selection_allowed": False,
        "holdout_skill_revision_allowed": False,
    }

    write_json_new(OUT_LOCK, lock)

    print("\n========================================")
    print("CREATED")
    print("========================================")

    print("\nDevelopment Fold A:")
    print("  total:", len(rows_a))
    print(
        "  formats:",
        dict(sorted(Counter(x["answer_format"] for x in rows_a).items()))
    )

    print("\nDevelopment Fold B:")
    print("  total:", len(rows_b))
    print(
        "  formats:",
        dict(sorted(Counter(x["answer_format"] for x in rows_b).items()))
    )

    print("\nFrozen Holdout:")
    print("  total:", len(rows_h))
    print(
        "  formats:",
        dict(sorted(Counter(x["answer_format"] for x in rows_h).items()))
    )

    print("\nSHA256:")
    print("  Fold A :", sha256(OUT_A))
    print("  Fold B :", sha256(OUT_B))
    print("  Holdout:", sha256(OUT_H))

    remaining_counts = Counter(
        x.answer_format.value
        for x in remaining_after_dev
    )

    print("\nRemaining reserve after this protocol:")
    for fmt in sorted(required):
        print(
            f"  {fmt}:",
            remaining_counts.get(fmt, 0)
        )

    print("\nIMPORTANT:")
    print("Fold A / Fold B may be used for Skill Evolution.")
    print("Holdout is frozen and must NEVER be used for rule generation or revision.")


if __name__ == "__main__":
    main()
