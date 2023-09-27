from datasets import load_dataset


def to_llama_2_prompt_format(input):

    llama_2_prompt_format =["<s>[INST] <<SYS>><</SYS>>\n{} [/INST] {} </s>".format(instruction,response) for instruction,response in zip(input["instruction"],input["output"])]
    return {"prompt": llama_2_prompt_format}
def load_open_platypus_dataset():
    dataset = load_dataset("garage-bAInd/Open-Platypus",split="train")

