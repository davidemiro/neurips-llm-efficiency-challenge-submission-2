from datasets import load_dataset, concatenate_datasets
from textattack.augmentation import WordNetAugmenter
from textattack.augmentation import CharSwapAugmenter

system_prompt = "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.Given a document your task is to generate summary based solely on the information presented in the input document."
task = "Summarize the given document."
wnattach = WordNetAugmenter()
chattach = CharSwapAugmenter()


def text_attack(batch):
    """
    :param input: A row of the multi news summarize
    :return: prompt SFT text attach
    """


    instructions = batch["instruction"]
    responses = batch["response"]

    attach_instructions = []
    attach_responses = []

    for instruction,response in zip(instructions,responses):

        #extract question to attach
        question = instruction[len(system_prompt)+1:]
        question = question[len(task)+1:]

        question = wnattach.augment(question)[0]
        question = chattach.augment(question)[0]

        attach_instructions.append("{}\n{}\n{}\n".format(system_prompt, task, question))
        attach_responses.append(response)

    return {"instruction": attach_instructions, "response": attach_responses}
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
    dataset = load_dataset("bigscience/P3", "multi_news_summarize", split="train[:3000]")

    dataset = dataset.map(sft_format,batched=True);
    dataset_attach1 = dataset.map(text_attack,batched=True)

    return concatenate_datasets([dataset,dataset_attach1])


