"""
Draft candidate labels for new, unlabeled queries using a local Ollama model.
You MUST manually review and correct every line before merging into train.jsonl —
this script only speeds up drafting, it does not produce trusted labels.

Usage:
    1. Put one query per line in new_queries.txt
    2. Run: python bootstrap_data.py --input new_queries.txt --output data/candidates.jsonl
    3. Open data/candidates.jsonl, review/fix every "completion" field
    4. Manually append the corrected lines to data/train.jsonl

Requires: ollama running locally (`ollama serve`) with a model pulled, e.g.:
    ollama pull llama3.1:8b
"""

import argparse
import json

import requests

RUBRIC_PROMPT = (
    "Classify the following user query as SIMPLE or COMPLEX for routing purposes.\n\n"
    "SIMPLE: factual lookups, single-step questions, greetings, basic calculations, "
    "straightforward requests with one clear intent.\n"
    "COMPLEX: multi-step reasoning, requests requiring synthesis across multiple facts, "
    "ambiguous or open-ended questions, tasks requiring planning, code generation, or "
    "comparing multiple options.\n\n"
    'Query: "{query}"\nClassification:'
)


def draft_label(query: str, model: str, ollama_url: str) -> str:
    prompt = RUBRIC_PROMPT.format(query=query)
    response = requests.post(
        f"{ollama_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": 5}},
        timeout=60,
    )
    response.raise_for_status()
    text = response.json().get("response", "").strip().upper()
    if "COMPLEX" in text:
        return "COMPLEX"
    if "SIMPLE" in text:
        return "SIMPLE"
    return "UNLABELED"  # forces you to look at it manually — never silently guess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Text file, one query per line")
    parser.add_argument("--output", default="data/candidates.jsonl")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model tag")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        queries = [line.strip() for line in f if line.strip()]

    print(f"Drafting labels for {len(queries)} queries using {args.model}...")
    print("REMINDER: every label below is a DRAFT. Review and correct all of them.\n")

    with open(args.output, "w") as out:
        for i, query in enumerate(queries, 1):
            label = draft_label(query, args.model, args.ollama_url)
            record = {
                "prompt": RUBRIC_PROMPT.format(query=query),
                "completion": f" {label}",
            }
            out.write(json.dumps(record) + "\n")
            print(f"  [{i}/{len(queries)}] {label:>10}  {query[:60]}")

    print(f"\nWrote draft labels to {args.output}.")
    print("Next: open that file, correct every label, then merge into data/train.jsonl.")


if __name__ == "__main__":
    main()
