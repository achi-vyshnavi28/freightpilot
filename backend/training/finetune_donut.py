"""Fine-tune Donut (OCR-free document understanding) on rendered + real document images.

Needs a GPU and `pip install ".[train]"`. NOT run in CI. Typical budget: a single
T4 / free Colab GPU, ~1 hour for 1k images at 960x720.

    python -m training.make_donut_dataset --out data/donut --n 400
    python -m training.finetune_donut --data data/donut --out models/donut-freight
    FP_DONUT_MODEL=models/donut-freight uvicorn app.main:app   # enables the "ml" extraction tier

Report field-level accuracy from evals on the validation split against the
rules and LLM tiers before switching it on; a model that is cheaper but wrong
is not a win.
"""

from __future__ import annotations

import argparse
import json

import torch
from datasets import load_dataset
from transformers import (
    DonutProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
)

BASE = "naver-clova-ix/donut-base"
DOC_TYPES = ["gst_invoice", "eway_bill", "lorry_receipt"]


def json_to_tokens(obj, new_tokens: set[str]) -> str:
    """{"invoice_no": "A/1"} -> "<s_invoice_no>A/1</s_invoice_no>" (Donut's target format)."""
    if isinstance(obj, dict):
        out = ""
        for key, value in obj.items():
            new_tokens.update({f"<s_{key}>", f"</s_{key}>"})
            out += f"<s_{key}>{json_to_tokens(value, new_tokens)}</s_{key}>"
        return out
    if isinstance(obj, list):
        return "<sep/>".join(json_to_tokens(v, new_tokens) for v in obj)
    return str(obj)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/donut")
    ap.add_argument("--out", default="models/donut-freight")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=384)
    args = ap.parse_args()

    processor = DonutProcessor.from_pretrained(BASE)
    processor.image_processor.size = {"height": 960, "width": 720}
    processor.image_processor.do_align_long_axis = False
    model = VisionEncoderDecoderModel.from_pretrained(BASE)

    ds = load_dataset("imagefolder", data_dir=args.data)
    new_tokens: set[str] = {"<sep/>"} | {f"<s_{t}>" for t in DOC_TYPES}
    targets = {}
    for split in ds:
        targets[split] = []
        for row in ds[split]:
            gt = json.loads(row["ground_truth"])["gt_parse"]
            doc_type = gt.pop("doc_type")
            # Task prompt token first, so inference can ask for one document type.
            targets[split].append(f"<s_{doc_type}>" + json_to_tokens(gt, new_tokens) + processor.tokenizer.eos_token)
    processor.tokenizer.add_special_tokens({"additional_special_tokens": sorted(new_tokens)})
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_gst_invoice>")

    def encode(split):
        def fn(batch, idx):
            pixel = processor(batch["image"], return_tensors="pt").pixel_values
            labels = processor.tokenizer(
                [targets[split][i] for i in idx],
                add_special_tokens=False,
                max_length=args.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids
            labels[labels == processor.tokenizer.pad_token_id] = -100  # ignore padding in the loss
            return {"pixel_values": pixel, "labels": labels}

        return ds[split].map(fn, batched=True, batch_size=8, with_indices=True, remove_columns=ds[split].column_names)

    train, val = encode("train"), encode("validation")
    train.set_format("torch")
    val.set_format("torch")

    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=3e-5,
            warmup_ratio=0.05,
            fp16=torch.cuda.is_available(),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            logging_steps=20,
            report_to="none",
        ),
        train_dataset=train,
        eval_dataset=val,
    )
    trainer.train()
    trainer.save_model(args.out)
    processor.save_pretrained(args.out)


if __name__ == "__main__":
    main()
