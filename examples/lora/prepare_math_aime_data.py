#!/usr/bin/env python3
"""
Prepare MATH (level 3-5) training data and AIME 2025 evaluation data.

Training: MATH competition problems (Hendrycks) filtered to difficulty levels 3-5
Evaluation: AIME 2025 (30 problems, integer answers 0-999)

Output format: JSONL with 'messages' and 'label' fields, compatible with miles RL training.
"""

import json
import os
import re
import sys

from datasets import load_dataset


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from \\boxed{...} in a solution string."""
    # Find the last \boxed{...} in the solution
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, solution)
    if matches:
        return matches[-1].strip()
    return ""


def prepare_math_training_data(output_path: str, min_level: int = 3):
    """Download MATH dataset, filter by level, and save as JSONL."""
    print(f"Downloading MATH dataset...")
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    print(f"  Total training problems: {len(ds)}")

    # Also get test split for more data
    ds_test = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")
    print(f"  Total test problems: {len(ds_test)}")

    records = []
    for split_name, split_ds in [("train", ds), ("test", ds_test)]:
        for row in split_ds:
            # Filter by level
            level_str = row.get("level", "")
            level_match = re.search(r"(\d+)", level_str)
            if not level_match:
                continue
            level = int(level_match.group(1))
            if level < min_level:
                continue

            problem = row["problem"]
            solution = row["solution"]
            answer = extract_boxed_answer(solution)
            if not answer:
                continue

            # Format as chat messages
            record = {
                "messages": [
                    {
                        "role": "user",
                        "content": problem + "\n\nPlease think step by step and put your final answer within \\boxed{}.",
                    }
                ],
                "label": answer,
                "level": level,
                "type": row.get("type", ""),
                "split": split_name,
            }
            records.append(record)

    print(f"  Level >= {min_level}: {len(records)} problems")

    # Save as JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved to: {output_path}")

    # Print level distribution
    from collections import Counter
    level_dist = Counter(r["level"] for r in records)
    for lvl in sorted(level_dist):
        print(f"    Level {lvl}: {level_dist[lvl]}")

    return records


def prepare_aime_eval_data(output_path: str):
    """Download AIME 2025 dataset and save as JSONL."""
    print(f"Downloading AIME 2025 dataset...")

    try:
        ds = load_dataset("yentinglin/aime_2025", split="train")
    except Exception:
        try:
            ds = load_dataset("MathArena/aime_2025", split="train")
        except Exception:
            ds = load_dataset("math-ai/aime25", split="train")

    print(f"  Total AIME 2025 problems: {len(ds)}")

    # Print column names for debugging
    print(f"  Columns: {ds.column_names}")

    records = []
    for i, row in enumerate(ds):
        # Different datasets may have different column names
        problem = row.get("problem", row.get("Problem", row.get("question", "")))
        answer = row.get("answer", row.get("Answer", row.get("ground_truth", "")))

        if not problem or answer is None:
            print(f"  WARNING: Skipping row {i} - missing problem or answer")
            continue

        # AIME answers are integers 0-999
        answer = str(answer).strip()

        record = {
            "messages": [
                {
                    "role": "user",
                    "content": problem
                    + "\n\nPlease think step by step and put your final answer within \\boxed{}.",
                }
            ],
            "label": answer,
            "problem_index": i + 1,
        }
        records.append(record)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved to: {output_path}")
    print(f"  Sample answers: {[r['label'] for r in records[:5]]}")

    return records


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/root/data"

    print("=" * 60)
    print("Preparing MATH + AIME 2025 data for LoRA RL training")
    print("=" * 60)

    # Prepare training data (MATH level 3-5)
    math_path = os.path.join(data_dir, "math_level3to5", "train.jsonl")
    prepare_math_training_data(math_path, min_level=3)

    print()

    # Prepare eval data (AIME 2025)
    aime_path = os.path.join(data_dir, "aime2025", "aime2025.jsonl")
    prepare_aime_eval_data(aime_path)

    print()
    print("=" * 60)
    print("Done! Data is ready for training.")
    print(f"  Training: {math_path}")
    print(f"  Eval:     {aime_path}")
    print("=" * 60)
