"""Unsloth QLoRA training for the Rocky LoRA.

Run inside WSL2 with CUDA visible (see finetune/README.md). This script is an offline
path: it never touches the running app. It is import-guarded so it fails with a clear
message (rather than a traceback) on a machine without Unsloth/CUDA - which is expected
on the Windows dev box; training happens in WSL2.

  python finetune/train.py --data finetune/data/rocky_dialogues.jsonl --out outputs/rocky-lora
"""

from __future__ import annotations

import argparse

BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Rocky LoRA (Unsloth QLoRA)")
    parser.add_argument("--data", default="finetune/data/rocky_dialogues.jsonl")
    parser.add_argument("--out", default="outputs/rocky-lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from trl import SFTTrainer, SFTConfig
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise SystemExit(
            "Unsloth/TRL/datasets not installed. Training runs in WSL2 with CUDA; "
            "see finetune/README.md. (" + str(exc) + ")"
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    dataset = load_dataset("json", data_files=args.data, split="train")

    def format_chat(row):
        return {"text": tokenizer.apply_chat_template(row["messages"], tokenize=False)}

    dataset = dataset.map(format_chat)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_ratio=0.05,
            logging_steps=5,
            seed=args.seed,
            dataset_text_field="text",
        ),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"LoRA saved to {args.out}")
    print("Next: merge + convert to GGUF Q5_K_M, then `ollama create rocky -f finetune/Modelfile`.")


if __name__ == "__main__":
    main()
