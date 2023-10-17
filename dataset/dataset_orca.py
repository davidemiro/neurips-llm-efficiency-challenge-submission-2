from datasets import load_dataset
import string
import random
def sft_format(batch):

    """
    :param input: A row of the Open-Orca dataset
    :return: prompt SFT
    """

    instructions = ["{}\n{}".format(system_prompt,question) for system_prompt,question in zip(batch["system_prompt"],batch["question"])]
    responses = ["{}".format(response) for response in batch["response"]]

    return {"instruction" : instructions, "response": responses}

def load_orca():
    dataset = load_dataset("Open-Orca/OpenOrca", split="train[:120000]")
    return dataset.map(sft_format, batched=True)

