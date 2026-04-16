#!/usr/bin/env python3
"""
Build the MedEval benchmark CSV from the DDXPlus English release.

Each DDXPlus patient carries a differential diagnosis — a distribution over
pathologies that is the analytical ground truth for this domain. For each
sampled patient we emit a question that presents the patient's age, sex, and
symptom list (translated from E-codes via release_evidences.json) and asks the
model for its single most likely diagnosis plus a confidence in [0, 1].

Columns written (matches BayesEval convention):
    question_id, question_prompt, confidence_prompt, age, sex,
    true_pathology, differential_json

true_probability is computed at scoring time as differential[model_answer],
mirroring LifeEval where the true probability depends on the model's guess.

Usage:
    python build_benchmark.py --ddxplus ../../../ddxplus/22687585 --n 500
"""

import argparse
import ast
import csv
import json
import random
import zipfile
from pathlib import Path


PATHOLOGY_INSTRUCTION = (
    "You are a diagnostic reasoning assistant. Based on the patient vignette "
    "below, give your single most likely pathology. You MUST commit to one "
    "diagnosis — do not hedge or list alternatives.\n\n{vignette}"
)

CONFIDENCE_INSTRUCTION = (
    "How confident are you (0 to 1) that your diagnosis is the exact pathology "
    "recorded for this patient? Respond with ONLY valid JSON: "
    '{"Answer": "<pathology>", "Confidence": "0.XX"}'
)


def load_evidence_map(evidences_path: Path) -> dict:
    with open(evidences_path) as f:
        return json.load(f)


def render_evidence(code: str, emap: dict) -> str:
    """Translate an EVIDENCES entry (possibly E_x_@_V_y or E_x_@_n) to English."""
    if "_@_" in code:
        ecode, value = code.split("_@_", 1)
    else:
        ecode, value = code, None
    entry = emap.get(ecode)
    if entry is None:
        return code
    question = entry.get("question_en", ecode)
    if value is None:
        return f"- {question}: yes"
    vm = entry.get("value_meaning") or {}
    meaning = vm.get(value, {}).get("en", value) if isinstance(vm, dict) else value
    return f"- {question} {meaning}"


def build_vignette(age, sex, evidences, emap) -> str:
    sex_word = {"M": "male", "F": "female"}.get(sex, sex)
    lines = [
        f"Patient: {age}-year-old {sex_word}.",
        "Reported findings:",
    ]
    lines.extend(render_evidence(c, emap) for c in evidences)
    return "\n".join(lines)


def iter_patients(patients_zip: Path):
    with zipfile.ZipFile(patients_zip) as zf:
        inner = next(n for n in zf.namelist() if not n.endswith("/"))
        with zf.open(inner) as f:
            text = f.read().decode("utf-8").splitlines()
    reader = csv.DictReader(text)
    for row in reader:
        yield row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddxplus", type=Path,
                    default=Path(__file__).parent / "Data",
                    help="Directory containing release_evidences.json and release_<split>_patients.zip")
    ap.add_argument("--split", default="test",
                    choices=["train", "validate", "test"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "Data" / "benchmark.csv")
    args = ap.parse_args()

    emap = load_evidence_map(args.ddxplus / "release_evidences.json")
    patients_zip = args.ddxplus / f"release_{args.split}_patients.zip"

    rng = random.Random(args.seed)
    pool = list(iter_patients(patients_zip))
    rng.shuffle(pool)
    pool = pool[: args.n]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "question_id", "question_prompt", "confidence_prompt",
            "age", "sex", "true_pathology", "differential_json",
        ])
        for i, p in enumerate(pool):
            evidences = ast.literal_eval(p["EVIDENCES"])
            differential = ast.literal_eval(p["DIFFERENTIAL_DIAGNOSIS"])
            vignette = build_vignette(p["AGE"], p["SEX"], evidences, emap)
            w.writerow([
                f"med_{args.split}_{i:05d}",
                PATHOLOGY_INSTRUCTION.format(vignette=vignette),
                CONFIDENCE_INSTRUCTION,
                p["AGE"], p["SEX"], p["PATHOLOGY"],
                json.dumps(differential),
            ])
    print(f"Wrote {len(pool)} questions to {args.out}")


if __name__ == "__main__":
    main()
