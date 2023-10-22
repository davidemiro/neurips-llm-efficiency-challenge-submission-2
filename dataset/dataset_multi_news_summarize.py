from datasets import load_dataset, concatenate_datasets
from textattack.augmentation import WordNetAugmenter
from textattack.augmentation import CharSwapAugmenter
import multiprocessing

system_prompt = "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.Given a document your task is to generate summary based solely on the information presented in the input document."
task = "Summarize the given document."

def sft_format(batch):

    """
    :param input: A row of the multi news summarize
    :return: prompt SFT
    """
    instructions = []
    responses = []
    inputs_pretokenized = batch["inputs_pretokenized"]
    targets_pretokenized = batch["targets_pretokenized"]

    for input_pretokenized,target_pretokenized in zip(inputs_pretokenized,targets_pretokenized):
        splits = input_pretokenized.split("\n")
        instruction = ""
        for s in splits:
            s = s.lstrip()
            if s == "":
                continue
            #Document: {content1} Document: {content2} --> {content1}\n{content2}
            if s.startswith("Document:"):
                continue
            #articles -> article
            if s == "Write a summary of the following articles:":
                continue
            instruction = instruction + s + "\n"

        response = "".join([s for s in target_pretokenized.split("\n")])

        instruction = instruction.replace("\\", "")
        response = response.replace("\\", "")

        system_prompt ="Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.Given a document your task is to generate summary based solely on the information presented in the input document."
        instruction = system_prompt+"\n"+task+"\nDocument: {" + instruction[:-1] + "}"
        response = "Summary: {" + response + "}"


        instructions.append(instruction)
        responses.append(response)

    return {"instruction" : instructions, "response": responses}

def load_multi_news_summarize():
    #multi_news_summarize
    # keys 'inputs', 'inputs_pretokenized', 'targets', 'targets_pretokenized'
    dataset = load_dataset("bigscience/P3", "multi_news_summarize", split="train[:6000]")

    dataset = dataset.map(sft_format,batched=True,remove_columns=['inputs', 'inputs_pretokenized', 'targets', 'targets_pretokenized'])
    return dataset



