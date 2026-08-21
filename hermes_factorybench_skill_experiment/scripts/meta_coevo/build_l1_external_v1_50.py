#!/usr/bin/env python3
"""
Build FactoryBench Level-1 External Test v1
===========================================

Purpose
-------
Create a FROZEN untouched external test set for the already-frozen Arm-F Skill.

Selection:
- Level 1 only
- split=test
- 20 four_letter_tf
- 30 scalar_range
- total 50 cases
- exclude every ID already used by:
    * dev fold A
    * dev fold B
    * original holdout
- deterministic metadata-diversity selection
- NO model calls
- NO GT/performance-based selection

This script refuses to overwrite an existing external manifest.
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

FACTORYBENCH_ROOT = Path(
    "/home/training/automatic_prompt_engineer/FactoryBench"
)

MANIFEST_DIR = ROOT / "data_manifests/meta_m1"

OUTPUT = MANIFEST_DIR / "factorybench_l1_external_test_v1.json"
AUDIT_OUTPUT = MANIFEST_DIR / "factorybench_l1_external_test_v1_audit.json"

OLD_MANIFESTS = [
    MANIFEST_DIR / "factorybench_l123_dev_fold_a.json",
    MANIFEST_DIR / "factorybench_l123_dev_fold_b.json",
    MANIFEST_DIR / "factorybench_l123_holdout.json",
]

TARGETS = {
    "four_letter_tf": 20,
    "scalar_range": 30,
}

# This is NOT a random ML seed. It is only used to give deterministic
# tie-breaking among equally diverse metadata candidates.
SELECTION_SEED = "factorybench-l1-external-v1-50cases-20260821"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_str(value: Any) -> str:
    return "unknown" if value is None else str(value)


def main() -> None:
    # ------------------------------------------------------------------
    # Imports: use the exact local FactoryBench + revision already used
    # by the existing experiment.
    # ------------------------------------------------------------------
    sys.path.insert(0, str(FACTORYBENCH_ROOT))
    from factorybench import load_split

    sys.path.insert(0, str(ROOT))
    from scripts.meta_coevo import run_static_surrogate as base

    # ------------------------------------------------------------------
    # Never silently replace a frozen external set.
    # ------------------------------------------------------------------
    if OUTPUT.exists() or AUDIT_OUTPUT.exists():
        raise RuntimeError(
            "External Test v1 already exists.\n"
            f"  manifest: {OUTPUT}\n"
            f"  audit   : {AUDIT_OUTPUT}\n\n"
            "Do NOT overwrite an external test set after it has been created."
        )

    # ------------------------------------------------------------------
    # 1. Collect all previously used IDs.
    # ------------------------------------------------------------------
    used_ids: set[str] = set()
    used_by_manifest: dict[str, list[str]] = {}

    for path in OLD_MANIFESTS:
        if not path.exists():
            raise FileNotFoundError(path)

        payload = load_json(path)
        ids = [row["id"] for row in payload["items"]]
        used_by_manifest[path.name] = ids
        used_ids.update(ids)

    print("========================================")
    print("FACTORYBENCH L1 EXTERNAL TEST V1 BUILDER")
    print("========================================")
    print("Dataset revision:", base.REV)
    print("Previously used unique IDs:", len(used_ids))
    print()

    # ------------------------------------------------------------------
    # 2. Load complete L1 test split.
    # train/validation are intentionally NOT used because this revision
    # showed they do not exist for L1.
    # ------------------------------------------------------------------
    items = load_split(
        1,
        split="test",
        revision=base.REV,
        max_items=None,
    )

    print("L1 test total:", len(items))

    # ------------------------------------------------------------------
    # 3. Remove all prior development/holdout cases.
    # ------------------------------------------------------------------
    unseen = [item for item in items if item.id not in used_ids]

    print("L1 unseen after exclusion:", len(unseen))
    print("\nUnseen format counts:")

    unseen_format_counts = Counter(
        item.answer_format.value for item in unseen
    )
    for fmt, count in sorted(unseen_format_counts.items()):
        print(f"  {fmt}: {count}")

    for fmt, needed in TARGETS.items():
        available = unseen_format_counts.get(fmt, 0)
        if available < needed:
            raise RuntimeError(
                f"Not enough unseen {fmt}: need={needed}, available={available}"
            )

    # ------------------------------------------------------------------
    # Metadata helpers.
    # ------------------------------------------------------------------
    def dataset_of(item):
        return safe_str(getattr(item, "dataset", None))

    def episode_of(item):
        provenance = getattr(item, "provenance", None) or {}
        return provenance.get("episode")

    def template_of(item):
        return getattr(item, "template_id", None)

    def tie_hash(item):
        raw = f"{SELECTION_SEED}|{item.id}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    # ------------------------------------------------------------------
    # 4. Deterministic diversity-aware selection.
    #
    # Important:
    # We use metadata only. We never inspect model outputs or GT scores.
    # Priority:
    #   1. new dataset
    #   2. new template
    #   3. new episode
    # Tie-break:
    #   stable SHA256(seed + item_id)
    # ------------------------------------------------------------------
    def select_diverse(pool, n):
        selected = []
        remaining = list(pool)

        seen_datasets = set()
        seen_templates = set()
        seen_episodes = set()

        while remaining and len(selected) < n:
            ranked = []

            for item in remaining:
                ds = dataset_of(item)
                tp = template_of(item)
                ep = episode_of(item)

                novelty = (
                    int(ds not in seen_datasets),
                    int(tp not in seen_templates),
                    int(ep not in seen_episodes),
                )

                ranked.append(
                    (
                        -novelty[0],
                        -novelty[1],
                        -novelty[2],
                        tie_hash(item),
                        item,
                    )
                )

            ranked.sort(key=lambda row: row[:4])
            item = ranked[0][4]

            selected.append(item)
            seen_datasets.add(dataset_of(item))
            seen_templates.add(template_of(item))
            seen_episodes.add(episode_of(item))

            remaining = [x for x in remaining if x.id != item.id]

        if len(selected) != n:
            raise RuntimeError(
                f"Selection failed: requested={n}, selected={len(selected)}"
            )

        return selected

    selected = []

    print("\nSelection plan:")
    for fmt, n in TARGETS.items():
        pool = [
            item for item in unseen
            if item.answer_format.value == fmt
        ]

        chosen = select_diverse(pool, n)
        selected.extend(chosen)

        print(f"  {fmt}: {n} / {len(pool)} unseen cases")

    # ------------------------------------------------------------------
    # 5. Build manifest rows expected by source_items().
    # ------------------------------------------------------------------
    rows = []

    for item in selected:
        rows.append(
            {
                "id": item.id,
                "level": 1,
                "split": "test",
                "episode": episode_of(item),
                "answer_format": item.answer_format.value,

                # Audit-only metadata; source_items() can ignore these.
                "dataset": getattr(item, "dataset", None),
                "template_id": template_of(item),
            }
        )

    external_ids = {row["id"] for row in rows}

    if len(external_ids) != len(rows):
        raise RuntimeError("Duplicate IDs inside external manifest")

    overlap = external_ids & used_ids
    if overlap:
        raise RuntimeError(
            f"DATA LEAKAGE: {len(overlap)} external IDs appeared previously"
        )

    # ------------------------------------------------------------------
    # 6. Reserve-pool accounting BEFORE writing.
    # ------------------------------------------------------------------
    selected_by_format = Counter(row["answer_format"] for row in rows)

    reserve_counts = {
        fmt: unseen_format_counts.get(fmt, 0) - selected_by_format.get(fmt, 0)
        for fmt in TARGETS
    }

    # ------------------------------------------------------------------
    # 7. Freeze manifest.
    # ------------------------------------------------------------------
    manifest = {
        "name": "factorybench_l1_external_test_v1",
        "purpose": (
            "Untouched same-level external generalization test for the "
            "already-frozen Arm-F Skill"
        ),
        "dataset_revision": base.REV,
        "frozen_after_creation": True,
        "selection_policy": {
            "level": 1,
            "split": "test",
            "exclude_prior_dev_and_holdout_ids": True,
            "prior_manifest_names": [p.name for p in OLD_MANIFESTS],
            "active_formats_only": list(TARGETS),
            "target_counts": TARGETS,
            "total_target_count": sum(TARGETS.values()),
            "selection_method": (
                "deterministic metadata-diversity selection; "
                "no agent outputs, GT scores, or failure labels used"
            ),
            "diversity_priority": [
                "dataset",
                "template_id",
                "episode",
            ],
            "tie_break_seed": SELECTION_SEED,
        },
        "items": rows,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest_sha = sha256(OUTPUT)

    # ------------------------------------------------------------------
    # 8. Audit payload.
    # ------------------------------------------------------------------
    selected_dataset_counts = Counter(
        safe_str(row.get("dataset")) for row in rows
    )
    selected_template_counts = Counter(
        safe_str(row.get("template_id")) for row in rows
    )
    selected_episode_counts = Counter(
        safe_str(row.get("episode")) for row in rows
    )

    audit = {
        "manifest": str(OUTPUT),
        "manifest_sha256": manifest_sha,
        "dataset_revision": base.REV,
        "prior_used_unique_id_count": len(used_ids),
        "l1_test_total": len(items),
        "l1_unseen_total_before_external_selection": len(unseen),
        "unseen_format_counts_before_selection": dict(
            sorted(unseen_format_counts.items())
        ),
        "external_total": len(rows),
        "external_format_counts": dict(
            sorted(selected_by_format.items())
        ),
        "external_dataset_counts": dict(
            sorted(selected_dataset_counts.items())
        ),
        "external_template_counts": dict(
            sorted(selected_template_counts.items())
        ),
        "unique_episode_count": len(selected_episode_counts),
        "remaining_unseen_active_format_reserve": reserve_counts,
        "overlap_with_prior_data": 0,
        "leakage_audit_pass": True,
        "selection_used_model_outputs": False,
        "selection_used_gt_performance": False,
    }

    AUDIT_OUTPUT.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # 9. Console report.
    # ------------------------------------------------------------------
    print("\n========================================")
    print("EXTERNAL TEST V1 CREATED")
    print("========================================")
    print("Manifest:", OUTPUT)
    print("Audit   :", AUDIT_OUTPUT)
    print("Total   :", len(rows))

    print("\nBy answer format:")
    for key, value in sorted(selected_by_format.items()):
        print(f"  {key}: {value}")

    print("\nBy dataset:")
    for key, value in sorted(selected_dataset_counts.items()):
        print(f"  {key}: {value}")

    print("\nBy template:")
    for key, value in sorted(selected_template_counts.items()):
        print(f"  {key}: {value}")

    print("\nUnique episodes:", len(selected_episode_counts))

    print("\nRemaining unseen active-format reserve:")
    for fmt, count in reserve_counts.items():
        print(f"  {fmt}: {count}")

    print("\nLEAKAGE AUDIT: PASS")
    print("Overlap with Fold A / Fold B / Holdout = 0")

    print("\nManifest SHA256:")
    print(manifest_sha)

    print("\nIMPORTANT:")
    print("This manifest is now FROZEN.")
    print("Do not replace, remove, or add cases after observing External Test results.")


if __name__ == "__main__":
    main()
