"""LoRA fine-tune of a small open LLM to normalise messy field values.

Task: raw OCR'd value -> canonical value. Examples:
    "Hyd" -> "Hyderabad"            "TS 09 UB 452I" -> "TS09UB4521"
    "64,0O0.00/-" -> "64000.00"     "14 Oct 26" -> "2026-10-14"

Why bother when the rules already do this? Interview answer: compare them.
Run evals with rules vs this adapter and report accuracy, latency and cost per
1k fields. A 0.5B model on CPU that matches the rules on clean data and beats
them on long-tail spellings is worth it; one that does not is not.

Needs `pip install ".[train]"` and ideally a GPU. NOT run in CI.

    python -m training.finetune_lora --out models/normaliser-lora
"""

from __future__ import annotations

import argparse
import random
from datetime import date

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from evals.dataset import CITY_ALIAS, build_shipments, fmt_date, fmt_money, fmt_plate

BASE = "Qwen/Qwen2.5-0.5B-Instruct"


def make_pairs(n_shipments: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pairs = []
    for ship in build_shipments(n_shipments, seed=seed):
        f = ship.docs[0].expected_fields
        d = date.fromisoformat(f["invoice_date"])
        pairs += [
            {"kind": "city", "raw": rng.choice(CITY_ALIAS.get(f["origin"], [f["origin"]])), "canonical": f["origin"]},
            {
                "kind": "vehicle",
                "raw": fmt_plate(rng, f["vehicle_no"]).replace("1", rng.choice("1I")),
                "canonical": f["vehicle_no"],
            },
            {
                "kind": "amount",
                "raw": fmt_money(rng, f["total_amount"]).replace("0", rng.choice("0O"), 1),
                "canonical": f"{f['total_amount']:.2f}",
            },
            {"kind": "date", "raw": fmt_date(rng, d), "canonical": d.isoformat()},
        ]
    return pairs


def to_text(p: dict, eos: str) -> str:
    return f"Normalise the {p['kind']}.\nInput: {p['raw']}\nOutput: {p['canonical']}{eos}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/normaliser-lora")
    ap.add_argument("--shipments", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"
        ),
    )
    model.print_trainable_parameters()  # ~0.5% of weights train; the base model stays frozen

    pairs = make_pairs(args.shipments, seed=21)
    ds = Dataset.from_list([{"text": to_text(p, tok.eos_token)} for p in pairs]).train_test_split(test_size=0.05, seed=0)
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=96), batched=True, remove_columns=["text"])

    Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=16,
            learning_rate=2e-4,
            eval_strategy="epoch",
            logging_steps=25,
            report_to="none",
        ),
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    ).train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    main()
