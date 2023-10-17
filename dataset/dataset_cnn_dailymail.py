from datasets import load_dataset
import string
import random

def sft_format(batch):

    """
    :param batch: cnn_daily dataset
    :return: prompt for llama 2
    """
    instructions = ["Summarize the given document.\nDocument: {"+ document +"}" for document in batch["article"]]
    responses = ["Summary: {"+ summary +"}" for summary in batch["highlights"]]

    return {"instruction" : instructions, "response": responses}

def load():

    dataset = load_dataset("cnn_dailymail","1.0.0", split="train[:60000]")
    llama_2_prompt_format_datasets = dataset.map(sft_format, batched=True)
    return llama_2_prompt_format_datasets

load()