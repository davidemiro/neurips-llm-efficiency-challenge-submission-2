from datasets import load_dataset, concatenate_datasets
import string
import random
from textattack.augmentation import WordNetAugmenter
from textattack.augmentation import CharSwapAugmenter
import multiprocessing

system_prompt = "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity."
task = "Given a multiple-choice question with answer choices labeled alphabetically (A, B, C, D, E, etc.), your task is to provide the letter of the correct choice as the answer. You should select the letter corresponding to the correct answer choice based on its knowledge. Your task is to only provide the letter of the correct choice as the answer."
wnattach = WordNetAugmenter()
chattach = CharSwapAugmenter()
def text_attack(batch):
    """
    :param input: A row of the bigscience/P3 sciq_Multiple_Choice dataset
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
    :param input: A row of the bigscience/P3 sciq_Multiple_Choice dataset
    :return: prompt SFT
    """
    instructions = []
    responses = []
    inputs_pretokenized = batch["inputs_pretokenized"]
    targets_pretolenized = batch["targets_pretokenized"]

    for input_pretokenized,target_pretokenized in zip(inputs_pretokenized,targets_pretolenized):

        splits = input_pretokenized.split("\n")
        correct_choice = target_pretokenized.split("\n")[0]
        _, _, _, _, _, question, _, _, _ = splits[:9]

        question = "Question: {}\n".format(question[3:])


        response_letter = "A"
        splits = splits[9:]

        choices = [line[2:] for line in splits if line != "" and line.startswith("-")]
        random.shuffle(choices)

        choices = [letter+". "+choice for choice,letter in zip(choices,string.ascii_uppercase)]


        for choice in choices:
            if choice[3:] == correct_choice:
                response_letter = choice[0]

            question += "{}.\n".format(choice)
        question += "Answer: "
        instruction = "{}\n{}\n{}\n".format(system_prompt, task, question)
        responses.append("{}".format(response_letter))
        instructions.append(instruction)

    return {"instruction" : instructions, "response": responses}

def load_p3_multi_qa():
    # keys 'inputs', 'inputs_pretokenized', 'targets', 'targets_pretokenized'
    dataset = load_dataset("bigscience/P3", "sciq_Multiple_Choice", split="train[:3000]")

    dataset = dataset.map(sft_format, batched=True, remove_columns=['inputs', 'inputs_pretokenized', 'targets', 'targets_pretokenized','answer_choices'])
    dataset_attach1 = dataset.map(text_attack, batched=True,num_proc=multiprocessing.cpu_count())

    return concatenate_datasets([dataset, dataset_attach1])





