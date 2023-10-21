import os
import sys
sys.path.insert(0, "/content/neurips_llm_efficiency_challenge")
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from configs.training_configs import *
from transformers.integrations import HfDeepSpeedConfig
from dataset import dataset_dolly
from model import residual_dropout
from optimum.bettertransformer import BetterTransformer
from dataset.prompt_formatter import prompt_formatter_func



# Load base model
model = residual_dropout.AutoModelForCausalLMWithResidualDropout.from_pretrained(
    model_name,
    use_auth_token=access_token,
    torch_dtype=torch.bfloat16,
    device_map= device_map,
    use_flash_attention_2=True,
)



tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          use_auth_token=access_token
                                          )
tokenizer.pad_token = tokenizer.eos_token

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)


model.hf_device_map
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_auth_token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training



# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_dolly.load_dolly_dataset(),
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
    formatting_func=prompt_formatter_func
)

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)