#!/usr/bin/env python3
"""
Entrenamiento QLoRA/LoRA optimizado para Llama 3.2:1B-Instruct.

Dataset esperado: JSONL en formato chat:
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}

Mejoras frente al script original:
- Supervisa principalmente la respuesta del assistant, no todo el prompt.
- Evita entrenar sobre tokens de padding usando labels=-100.
- Usa padding dinámico para no desperdiciar memoria.
- Incluye split de validación opcional.
- Usa QLoRA 4-bit por defecto si hay GPU compatible.
- Activa gradient checkpointing y group_by_length para ahorrar VRAM.
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenar un adaptador LoRA para Tutor Analítico Híbrido con Llama 3.2:1B"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Modelo base de Hugging Face. Para Ollama llama3.2:1b usa el equivalente HF: meta-llama/Llama-3.2-1B-Instruct.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tutor.jsonl",
        help="Archivo JSONL en formato messages.",
    )
    parser.add_argument("--output_dir", type=str, default="./lora-tutor-violencia")

    # Para tu dataset actual de ~250 ejemplos conviene no pasarse de épocas.
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1.5e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)

    # LoRA. r=16 mejora capacidad frente a r=8 sin volverlo excesivo para 1B.
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument(
        "--quantization",
        choices=["4bit", "8bit", "none"],
        default="4bit",
        help="4bit recomendado para GPU con poca VRAM. Usa 'none' si entrenas sin bitsandbytes.",
    )
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument(
        "--train_on_full_conversation",
        action="store_true",
        help="Si se activa, calcula pérdida sobre system+user+assistant. Por defecto solo supervisa la respuesta del assistant.",
    )

    return parser.parse_args()


def validate_jsonl(path: str) -> None:
    """Validación ligera para detectar errores antes de cargar el entrenamiento."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el dataset: {path}")

    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido en línea {line_no}: {exc}") from exc

            messages = example.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"Línea {line_no}: falta 'messages' o tiene formato inválido.")

            roles = [m.get("role") for m in messages]
            if "assistant" not in roles:
                raise ValueError(f"Línea {line_no}: no contiene respuesta de assistant.")
            for msg in messages:
                if msg.get("role") not in {"system", "user", "assistant"}:
                    raise ValueError(f"Línea {line_no}: role inválido: {msg.get('role')}")
                if not isinstance(msg.get("content"), str) or not msg.get("content").strip():
                    raise ValueError(f"Línea {line_no}: content vacío o inválido.")

    print(f"Dataset validado correctamente: {total} ejemplos.")


def load_model_and_tokenizer(model_name: str, quantization: str):
    print(f"Cargando tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    has_cuda = torch.cuda.is_available()
    compute_dtype = torch.bfloat16 if has_cuda and torch.cuda.is_bf16_supported() else torch.float16

    quantization_config = None
    if has_cuda and quantization == "4bit":
        print("Cargando modelo en 4-bit QLoRA.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif has_cuda and quantization == "8bit":
        print("Cargando modelo en 8-bit.")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif not has_cuda:
        print("AVISO: No se detectó GPU. En CPU el entrenamiento será muy lento.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.float32 if not has_cuda else compute_dtype,
        device_map="auto" if has_cuda else None,
        attn_implementation="sdpa",
    )

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False
    return model, tokenizer


def apply_lora(model, r: int, alpha: int, dropout: float):
    print("Aplicando LoRA...")

    # Para Llama, estos módulos cubren atención y MLP.
    # Si quieres algo más ligero, deja solo q_proj,k_proj,v_proj,o_proj.
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def build_prompt_and_full_text(messages: List[Dict[str, str]], tokenizer):
    """Separa prompt y respuesta para poder enmascarar la pérdida del prompt."""
    assistant_index = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            assistant_index = i
            break

    if assistant_index is None:
        raise ValueError("El ejemplo no contiene mensaje de assistant.")

    prompt_messages = messages[:assistant_index]
    assistant_text = messages[assistant_index]["content"].strip()

    # Prompt con marcador de generación propio del modelo instruct.
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = prompt_text + assistant_text + tokenizer.eos_token
    return prompt_text, full_text


def prepare_dataset(dataset_path: str, tokenizer, max_length: int, eval_ratio: float, seed: int, train_on_full: bool):
    print(f"Cargando dataset: {dataset_path}")
    validate_jsonl(dataset_path)

    dataset = load_dataset("json", data_files=dataset_path)["train"]

    if eval_ratio and 0 < eval_ratio < 1 and len(dataset) >= 20:
        split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    def tokenize(example):
        messages = example["messages"]
        prompt_text, full_text = build_prompt_and_full_text(messages, tokenizer)

        full = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=False,
        )

        labels = full["input_ids"].copy()

        if not train_on_full:
            prompt = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_length,
                padding=False,
                add_special_tokens=False,
            )
            prompt_len = min(len(prompt["input_ids"]), len(labels))
            labels[:prompt_len] = [-100] * prompt_len

        full["labels"] = labels
        return full

    train_dataset = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(tokenize, remove_columns=eval_dataset.column_names)

    print(f"Ejemplos de entrenamiento: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"Ejemplos de validación: {len(eval_dataset)}")

    return train_dataset, eval_dataset


@dataclass
class DataCollatorForCausalLMWithLabelMask:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            remainder = max_len - len(label)
            padded_labels.append(label + [-100] * remainder)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model_name, args.quantization)
    model = apply_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)

    train_dataset, eval_dataset = prepare_dataset(
        args.dataset,
        tokenizer,
        args.max_length,
        args.eval_ratio,
        args.seed,
        args.train_on_full_conversation,
    )

    has_cuda = torch.cuda.is_available()
    use_bf16 = has_cuda and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.save_steps if eval_dataset is not None else None,
        fp16=has_cuda and not use_bf16,
        bf16=use_bf16,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit" if has_cuda and args.quantization in {"4bit", "8bit"} else "adamw_torch",
        gradient_checkpointing=True,
        group_by_length=True,
        report_to="none",
        remove_unused_columns=False,
        max_grad_norm=0.3,
    )

    data_collator = DataCollatorForCausalLMWithLabelMask(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("\n--- Iniciando entrenamiento LoRA optimizado ---")
    trainer.train()

    print(f"\nGuardando adaptadores LoRA en: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Entrenamiento completado.")


if __name__ == "__main__":
    main()
