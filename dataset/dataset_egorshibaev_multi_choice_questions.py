from datasets import load_dataset

def load_question():
    dataset = load_dataset("garage-bAInd/Open-Platypus",split="train")
    llama_2_prompt_format_datasets = dataset.map(to_llama_2_prompt_format, batched=True)
    return dataset

def to_llama_2_prompt_format(input):

    llama_2_prompt_format =["<s>[INST] <<SYS>><</SYS>>\n{} [/INST] {} </s>".format(instruction,response) for instruction,response in zip(input["instruction"],input["output"])]
    return {"prompt": llama_2_prompt_format}
