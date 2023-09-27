# Load datasets (you can process it here)
from datasets import load_dataset,concatenate_datasets




# Tokenize the datasets

# LLaMa 2 prompt
"""
<s>[INST] <<SYS>>
System Prompt: context for the LLM (optional)
<</SYS>>
User Prompt: Instruction (required) [/INST] Model answare (required) </s>

"""
def to_llama_2_prompt_format(input):

    llama_2_prompt_format =["<s>[INST] <<SYS>>\n{}\n<</SYS>>\n{} [/INST] {} </s>".format(context,instruction,response) for context,instruction,response in zip(input["context"],input["instruction"],input["response"])]
    return {"prompt": llama_2_prompt_format}


def load_dolly_dataset():
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    llama_2_prompt_format_datasets = dataset.map(to_llama_2_prompt_format, batched=True)
    return llama_2_prompt_format_datasets
