# The model that you want to train from the Hugging Face hub
model_name = "mistralai/Mistral-7B-v0.1"



# Fine-tuned model name
new_model = "Mistral-dm"

################################################################################
# QLoRA parameters
################################################################################

# LoRA rank dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"



# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = True
tf32 = False

# Batch size per GPU for training
per_device_train_batch_size = 1

# Batch size per GPU for evaluation
per_device_eval_batch_size = 1

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "adamw_hf"

# Learning rate schedule (constant a bit better than cosine)
lr_scheduler_type = "constant"

# Number of training epochs
num_train_epochs = 10

# Number of training steps (overrides num_train_epochs)
max_steps = 115000

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 30000

# Log every X updates steps
logging_steps = 100



################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = 2048

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}

#
################################################################################
# Hugging face access token
################################################################################
access_token = "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"
