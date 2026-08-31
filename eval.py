"""
Compare base vs. LoRA-fine-tuned model on the held-out validation set.

Usage:
    python eval.py
    python eval.py --adapter-path ./adapters --data data/valid.jsonl

Requires: pip install mlx-lm
"""

import argparse
import json
from pathlib import Path

from mlx_lm import load, generate


def load_examples(path: str):
    examples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def extract_label(text: str) -> str:
    """Pull SIMPLE or COMPLEX out of generated text, robust to extra tokens."""
    text_upper = text.strip().upper()
    if "COMPLEX" in text_upper:
        return "COMPLEX"
    if "SIMPLE" in text_upper:
        return "SIMPLE"
    return text_upper.split()[0] if text_upper.split() else "UNKNOWN"


def run_eval(model, tokenizer, examples, label: str):
    correct = 0
    rows = []
    for ex in examples:
        prompt = ex["prompt"]
        gold = ex["completion"].strip().upper()

        # Use chat template if the model expects one; fall back to raw prompt.
        if tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            formatted = prompt

        output = generate(
            model, tokenizer, prompt=formatted, max_tokens=5, verbose=False
        )
        predicted = extract_label(output)
        is_correct = predicted == gold
        correct += int(is_correct)

        rows.append(
            {
                "query": prompt.split('Query: "')[-1].split('"')[0],
                "gold": gold,
                "predicted": predicted,
                "correct": is_correct,
            }
        )

    accuracy = correct / len(examples) if examples else 0.0
    print(f"\n=== {label} — accuracy: {accuracy:.1%} ({correct}/{len(examples)}) ===")
    for r in rows:
        mark = "✓" if r["correct"] else "✗"
        print(f"  {mark} [{r['gold']:>7} -> {r['predicted']:>7}]  {r['query'][:70]}")
    return accuracy, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-path", default="./adapters")
    parser.add_argument("--data", default="data/valid.jsonl")
    args = parser.parse_args()

    examples = load_examples(args.data)

    print("Loading base model (no adapter)...")
    base_model, base_tokenizer = load(args.model)
    base_acc, base_rows = run_eval(base_model, base_tokenizer, examples, "BASE MODEL")

    if Path(args.adapter_path).exists():
        print("\nLoading fine-tuned model (with adapter)...")
        ft_model, ft_tokenizer = load(args.model, adapter_path=args.adapter_path)
        ft_acc, ft_rows = run_eval(
            ft_model, ft_tokenizer, examples, "FINE-TUNED MODEL"
        )

        print("\n=== SUMMARY ===")
        print(f"Base model accuracy:        {base_acc:.1%}")
        print(f"Fine-tuned model accuracy:  {ft_acc:.1%}")
        print(f"Delta:                      {ft_acc - base_acc:+.1%}")

        flipped_to_correct = [
            b["query"]
            for b, f in zip(base_rows, ft_rows)
            if not b["correct"] and f["correct"]
        ]
        flipped_to_wrong = [
            b["query"]
            for b, f in zip(base_rows, ft_rows)
            if b["correct"] and not f["correct"]
        ]
        if flipped_to_correct:
            print(f"\nFixed by fine-tuning ({len(flipped_to_correct)}):")
            for q in flipped_to_correct:
                print(f"  + {q[:70]}")
        if flipped_to_wrong:
            print(f"\nBroken by fine-tuning ({len(flipped_to_wrong)}):")
            for q in flipped_to_wrong:
                print(f"  - {q[:70]}")
    else:
        print(
            f"\nNo adapter found at {args.adapter_path} — run training first "
            "(see README step 3), then re-run this script."
        )


if __name__ == "__main__":
    main()
