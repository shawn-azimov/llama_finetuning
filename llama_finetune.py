# -*- coding: utf-8 -*-

import torch
import evaluate
import re
import json
import time

import wandb
import pandas as pd

from torchinfo import summary

from transformers import AutoTokenizer, AutoModelForCausalLM, TorchAoConfig, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from huggingface_hub import notebook_login
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig

from accelerate import Accelerator

from tqdm import tqdm
from collections import defaultdict

from peft import LoraConfig, get_peft_model, PeftModel

from torch.utils.tensorboard import SummaryWriter

import os
from dotenv import load_dotenv
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

def tokenize_finetuning_examples(examples, tokenizer):
    inputs = tokenizer(
        examples["input_text"],
        #padding="max_length",
        padding="longest",
        padding_side='left',
        truncation=False,
        #max_length=1200,
        return_tensors='pt'
    )

    label = tokenizer(
        examples["target_text"],
        padding=False,
        truncation=False,
        add_special_tokens=False,
        return_tensors='pt'
    )

    labels = torch.full(inputs["input_ids"].shape, -100)
    labels[:, -1] = label["input_ids"][:, 0]

    inputs["labels"] = labels

    # print(inputs["input_ids"].shape)
    # print(inputs["labels"].shape)

    return inputs


def load_model_and_tokenizer(model_name, load_in_float16=False, use_dynamic_quantization=False, quant_type='int8'):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingface_token)
    if load_in_float16:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda', torch_dtype=torch.float16, token=huggingface_token)
    else:
        if use_dynamic_quantization:
            if quant_type == 'int4':
                quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, token=huggingface_token)
            elif quant_type == 'int8':
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise Exception(f'Invalid quantization type {quant_type}')

            model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', quantization_config=quantization_config, token=huggingface_token)
            print(f"\nDynamic quantization {quant_type} applied to model.")

            # quantization_config = TorchAoConfig("int8_weight_only")
            # model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda', quantization_config=quantization_config, token=huggingface_token)
            # print("Dynamic quantization int8_weight_only applied to model.")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda', token=huggingface_token)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    print(model.base_model)
    memory_allocated = torch.cuda.memory_allocated() / 1024 ** 2
    memory_reserved = torch.cuda.memory_reserved() / 1024 ** 2
    print(f"\nMemory Allocated: {memory_allocated:.2f} MB")
    print(f"Memory Reserved: {memory_reserved:.2f} MB\n")
    
    return model, tokenizer


def load_csv_files(file_paths):
    combined_data = []
    for file_path in file_paths:
        data = pd.read_csv(file_path, header=None)
        for _, row in data.iterrows():
            question = row[0]
            options = [row[1], row[2], row[3], row[4]]
            answer = row[5]

            combined_data.append({
                "question": question,
                "options": options,
                "answer": answer
            })
    return Dataset.from_pandas(pd.DataFrame(combined_data))


def prepare_finetuning_data(dataset, hugging_face_dataset=False):
    def create_finetuning_examples(example):
        question = example['question']
        options = example['choices' if hugging_face_dataset else 'options']
        answer_label = example['answer']
        prompt = ""

        prompt += "<|start_header_id|>user<|end_header_id|>\n\n"
        prompt += "Given the following question and four candidate answers (A, B, C, and D), choose the best answer.\n"
        prompt += f"Question: {question}\n"
        option_labels = ['A', 'B', 'C', 'D']
        for label, option_text in zip(option_labels, options):
            prompt += f"{label}. {option_text}\n"
        prompt += 'Your response should end with "The best answer is [the_answer_letter]." where [the_answer_letter] is one of A, B, C, or D.'
        prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        prompt += "The best answer is"
        
        if hugging_face_dataset:
            answer_index = answer_label
            option_labels = ['A', 'B', 'C', 'D']
            if 0 <= answer_index < len(option_labels):
                answer_label = option_labels[answer_index]
            else:
                answer_label = "N/A"
        
        return {"input_text": prompt, "target_text": " " + answer_label}

    finetuning_dataset = dataset.map(create_finetuning_examples, remove_columns=dataset.column_names)
    return finetuning_dataset

def fine_tune_model_with_lora(model, tokenizer, lora_config, train_dataset, val_dataset, batch_size, model_save_dir):

    model = get_peft_model(model, lora_config)
    print("\nModel adapted with LoRA.")
    
    print(summary(model))
    print(model.base_model)
    
    print(f"batch_size {batch_size}")
    
    memory_allocated = torch.cuda.memory_allocated() / 1024 ** 2
    memory_reserved = torch.cuda.memory_reserved() / 1024 ** 2
    print(f"\nMemory Allocated: {memory_allocated:.2f} MB")
    print(f"Memory Reserved: {memory_reserved:.2f} MB\n")
    
    train_dataset = prepare_finetuning_data(train_dataset)#, hugging_face_dataset=True)
    val_dataset = prepare_finetuning_data(val_dataset, hugging_face_dataset=True)
    
    train_dataset = train_dataset.map(
        lambda examples: tokenize_finetuning_examples(examples, tokenizer),
        batched=True,
        batch_size=batch_size,
        remove_columns=["input_text", "target_text"]
    )

    val_dataset = val_dataset.map(
        lambda examples: tokenize_finetuning_examples(examples, tokenizer),
        batched=True,
        batch_size=batch_size,
        remove_columns=["input_text", "target_text"]
    )
    
    training_args = SFTConfig(
        torch_empty_cache_steps=100,
        output_dir=model_save_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        eval_strategy='steps',
        eval_steps=100,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        fp16=True,
        save_total_limit=3,
        report_to="wandb"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    trainer = SFTTrainer(
        max_seq_length=1200,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    print(f"Fine-tuning complete. Model saved to {model_save_dir}.")


def main():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    load_in_float16 = True
    use_dynamic_quantization = False
    quant_type = 'int4'
    batch_size = 2
    run_name = 'finetuned_model_with_lora_fp16'
    
    model_save_dir = f'./{run_name}'
    
    model, tokenizer = load_model_and_tokenizer(model_name,
                                                load_in_float16=load_in_float16,
                                                use_dynamic_quantization=use_dynamic_quantization,
                                                quant_type=quant_type)

    csv_files = [
                "./data/mmlu/auxillary_train/arc_easy.csv",
                "./data/mmlu/auxillary_train/arc_hard.csv",
                "./data/mmlu/auxillary_train/aux_law_90s.csv",
                "./data/mmlu/auxillary_train/mc_test.csv",
                "./data/mmlu/auxillary_train/obqa.csv",
              #   "./data/mmlu/auxillary_train/race.csv",
                "./data/mmlu/auxillary_train/science_elementary.csv",
                "./data/mmlu/auxillary_train/science_middle.csv"
                ]

    train_dataset = load_csv_files(csv_files)
    #train_dataset = load_dataset('cais/mmlu', 'all')['test']
    print(f"Loaded {len(train_dataset)} training samples.")

    val_dataset = load_dataset('cais/mmlu', 'all')['validation']
    print(f"Loaded {len(val_dataset)} validation samples.")

    lora_r = 16
    lora_alpha = 16
    lora_dropout = 0.1
    
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )
    
    wandb.init(
        project="fine_tuning_with_lora",
        name=run_name,
        config={
            "model_name": model_name,
            "batch_size": batch_size,
            "fp16": load_in_float16,
            "quant": use_dynamic_quantization,
            "quant_type": quant_type,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
        }
    )
    
    fine_tune_model_with_lora(model, tokenizer, lora_config, train_dataset, val_dataset, batch_size, model_save_dir)

    wandb.finish()

if __name__ == "__main__":
    main()
