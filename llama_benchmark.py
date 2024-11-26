# -*- coding: utf-8 -*-

import torch
import evaluate
import re
import json
import time

from transformers import AutoTokenizer, AutoModelForCausalLM, TorchAoConfig, BitsAndBytesConfig
from huggingface_hub import notebook_login
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict

from peft import PeftModel

from torch.utils.tensorboard import SummaryWriter

import os
from dotenv import load_dotenv
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

def load_model_and_tokenizer(model_name, load_in_float16=False, use_dynamic_quantization=False, quant_type='int8'):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=huggingface_token)
    if load_in_float16:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda', torch_dtype=torch.float16, token=huggingface_token)
    else:
        if use_dynamic_quantization:
            if quant_type == 'int4':
                quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            elif quant_type == 'int8':
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise Exception(f'Invalid quantization type {quant_type}')
                
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda', quantization_config=quantization_config, token=huggingface_token)
            print(f"Dynamic quantization {quant_type} applied to model.")
            
            # quantization_config = TorchAoConfig("int8_weight_only")
            # model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda', quantization_config=quantization_config, token=huggingface_token)
            # print("Dynamic quantization int8_weight_only applied to model.")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda', token=huggingface_token)

    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # print("\n\nLayer dtypes:")
    # for name, param in model.named_parameters():
    #     print(f"{name}: dtype={param.dtype}")
    print(model.base_model)
    
    return model, tokenizer


def load_finetuned_model_and_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cuda', torch_dtype=torch.float16, token=huggingface_token)
    model = PeftModel.from_pretrained(model, model_path)
    
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=huggingface_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Finetuned Model and tokenizer loaded successfully from {model_path}.")
    print(model.base_model)
    
    return model, tokenizer

def parse_mmlu_example(example):
    question = example['question'].strip()
    options = example['choices']
    answer_index = example['answer']
    subject = example['subject']

    option_labels = ['A', 'B', 'C', 'D']
    if 0 <= answer_index < len(option_labels):
        answer_label = option_labels[answer_index]
    else:
        answer_label = "N/A"

    return {
        'question': question,
        'options': options,
        'answer': answer_label,
        'subject': subject
    }


def load_and_parse_dataset(dataset_name, parse_function, split, subset=None, max_samples=None):
    if subset is None:
        dataset = load_dataset(dataset_name)[split]
    else:
        dataset = load_dataset(dataset_name, subset)[split]

    dataset = dataset.map(parse_function)

    if max_samples:
        dataset = dataset.select(range(max_samples))

    return dataset


def create_prompt(question, options, few_shot=False, examples=None):
    prompt = ""
    if few_shot and examples:
        for ex in examples:
            prompt += "<|start_header_id|>user<|end_header_id|>\n\n"
            prompt += "Given the following question and four candidate answers (A, B, C, and D), choose the best answer.\n"
            prompt += f"Question: {ex['question']}\n"
            option_labels = ['A', 'B', 'C', 'D']
            for label, option_text in zip(option_labels, ex['options']):
                prompt += f"{label}. {option_text}\n"
            prompt += 'Your response should end with "The best answer is [the_answer_letter]." where [the_answer_letter] is one of A, B, C, or D.\n'
            prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            prompt += f"The best answer is {ex['answer']}\n\n"

    prompt += "<|start_header_id|>user<|end_header_id|>\n\n"
    prompt += "Given the following question and four candidate answers (A, B, C, and D), choose the best answer.\n"
    prompt += f"Question: {question}\n"
    option_labels = ['A', 'B', 'C', 'D']
    for label, option_text in zip(option_labels, options):
        prompt += f"{label}. {option_text}\n"
    prompt += 'Your response should end with "The best answer is [the_answer_letter]" where [the_answer_letter] is one of A, B, C, or D.'
    prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    prompt += "The best answer is"
    return prompt


def extract_choice(output):
    match = re.search(r'The best answer is\s*([A-D])', output, re.IGNORECASE)

    # print("***")
    # print(output)
    # print("***\n")

    if match:
        return match.group(1).upper()
    else:
        # for option in ['A', 'B', 'C', 'D']:
        #     if option in output.upper():
        #         return option
        return "N/A"


def predict_single_example(model, tokenizer, question, options, few_shot=False, examples=None, max_new_tokens=5):
    prompt = create_prompt(question, options, few_shot, examples)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=False).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = extract_choice(decoded_output)
    return answer, prompt


def predict_batch(model, tokenizer, questions, options_list, few_shot=False, examples=None, max_new_tokens=5):
    prompts = [create_prompt(q, o, few_shot, examples) for q, o in zip(questions, options_list)]

    inputs = tokenizer(prompts, return_tensors='pt', padding=True, padding_side='left', truncation=False).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            #top_p=0.9,
            #temperature=1.5,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    answers = [extract_choice(output) for output in decoded_outputs]
    return answers, prompts


def compute_accuracies(predictions, labels, subjects):
    option_label_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    predictions_mapped = [option_label_to_index.get(pred, -1) for pred in predictions]

    subject_data = defaultdict(lambda: {'predictions': [], 'labels': []})

    for pred, label, subj in zip(predictions_mapped, labels, subjects):
        subject_data[subj]['predictions'].append(pred)
        subject_data[subj]['labels'].append(label)

    accuracy_metric = evaluate.load('accuracy')
    subject_accuracy = {}
    for subject in sorted(subject_data.keys()):
        data = subject_data[subject]
        accuracy = accuracy_metric.compute(predictions=data["predictions"], references=data["labels"])
        subject_accuracy[subject] = accuracy['accuracy']

    overall_accuracy = accuracy_metric.compute(predictions=predictions_mapped, references=labels)['accuracy']

    return overall_accuracy, subject_accuracy


def save_data(file_name, predictions, labels, subjects, overall_accuracy, subject_accuracy, time, avg_memory_allocated, avg_memory_reserved):

    results_data = {
        'overall_accuracy': overall_accuracy,
        'per_subject_accuracy': subject_accuracy,
        'per_sample': [
            {
                'Subject': subj,
                'Prediction': pred,
                'Label': label
            }
            for pred, label, subj in zip(predictions, labels, subjects)
        ],
        'total_time' : time,
        'average_memory_allocated_MB': avg_memory_allocated,
        'average_memory_reserved_MB': avg_memory_reserved
    }

    with open(file_name + '.json', 'w') as json_file:
        json.dump(results_data, json_file, indent=4)


def main():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    load_in_float16 = True
    use_dynamic_quantization = False
    quant_type = 'int8'
    results_file_name = 'Llama-3.2-1B-Instruct_mmlu_evaluation_results_fp16_test_finetuned'
    split = 'test'
    batch_size = 4
    
    load_finetuned_model = True
    finetuned_model_path = './finetuned_model_with_lora_fp16_hf_dataset'
    
    if load_finetuned_model:
        model, tokenizer = load_finetuned_model_and_tokenizer(finetuned_model_path)
    else:
        model, tokenizer = load_model_and_tokenizer(model_name,
                                                    load_in_float16=load_in_float16,
                                                    use_dynamic_quantization=use_dynamic_quantization,
                                                    quant_type=quant_type)

    dataset = load_and_parse_dataset('cais/mmlu', parse_function=parse_mmlu_example, split=split, subset='all', max_samples=None)

    few_shot = False
    if few_shot:
        few_shot_examples = [dataset[i] for i in range(min(3, len(dataset)))]
    else:
        few_shot_examples = None

    predictions = []
    labels = []
    subjects = []

    log_dir = f'runs/{results_file_name}'
    writer = SummaryWriter(log_dir=log_dir)
    
    start_time = time.time()

    total_memory_allocated = 0
    total_memory_reserved = 0
    num_memory_measurements = 0
    
    if batch_size == 1:
        for i in tqdm(range(len(dataset)), desc="Evaluating"):
            batch = dataset.select([i])
            question = batch['question'][0]
            options = batch['options'][0]

            predicted_answer, _ = predict_single_example(
                model=model,
                tokenizer=tokenizer,
                question=question,
                options=options,
                few_shot=few_shot,
                examples=few_shot_examples,
                max_new_tokens=5
            )

            memory_allocated = torch.cuda.memory_allocated() / 1024 ** 2
            memory_reserved = torch.cuda.memory_reserved() / 1024 ** 2
            writer.add_scalar("Memory Allocated", memory_allocated, i)
            writer.add_scalar("Memory Reserved", memory_reserved, i)

            total_memory_allocated += memory_allocated
            total_memory_reserved += memory_reserved
            num_memory_measurements += 1
            
            predictions.append(predicted_answer)
            labels.append(batch['answer'][0])
            subjects.append(batch['subject'][0])

            torch.cuda.empty_cache()

    else:
        print(f'Batch size {batch_size}')
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
            end_idx = min(i + batch_size, len(dataset))
            batch = dataset.select(range(i, end_idx))

            questions = batch['question']
            options_list = batch['options']

            predicted_answers, _ = predict_batch(
                model=model,
                tokenizer=tokenizer,
                questions=questions,
                options_list=options_list,
                few_shot=few_shot,
                examples=few_shot_examples,
                max_new_tokens=10
            )

            #print(predicted_answers)

            memory_allocated = torch.cuda.memory_allocated() / 1024 ** 2
            memory_reserved = torch.cuda.memory_reserved() / 1024 ** 2
            writer.add_scalar("Memory Allocated", memory_allocated, i)
            writer.add_scalar("Memory Reserved", memory_reserved, i)

            total_memory_allocated += memory_allocated
            total_memory_reserved += memory_reserved
            num_memory_measurements += 1
            
            predictions.extend(predicted_answers)
            labels.extend(batch['answer'])
            subjects.extend(batch['subject'])

            del batch, questions, options_list, predicted_answers
            torch.cuda.empty_cache()

    writer.close()

    end_time = time.time()
    total_time = end_time - start_time

    option_label_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    predictions_mapped = [option_label_to_index.get(pred, -1) for pred in predictions]

    unique_values = set(predictions_mapped)
    print("\nUnique values in predictions_mapped:", unique_values)

    overall_accuracy, subject_accuracy = compute_accuracies(predictions, labels, subjects)

    avg_memory_allocated = total_memory_allocated / num_memory_measurements
    avg_memory_reserved = total_memory_reserved / num_memory_measurements
    
    save_data(results_file_name, predictions_mapped, labels, subjects, overall_accuracy, subject_accuracy, total_time, avg_memory_allocated, avg_memory_reserved)

    print(f"\n\nAccuracy by subject:")
    for subject, accuracy in subject_accuracy.items():
        print(f"Subject: {subject}, Accuracy: {accuracy * 100:.2f}%")

    print(f"\nOverall Accuracy: {overall_accuracy * 100:.2f}%")
    
    print(f"\n\nAverage Memory Allocated: {avg_memory_allocated:.2f} MB")
    print(f"Average Memory Reserved: {avg_memory_reserved:.2f} MB")

    print(f"\nTotal time: {total_time:.2f} seconds")
    
if __name__ == "__main__":
    main()
