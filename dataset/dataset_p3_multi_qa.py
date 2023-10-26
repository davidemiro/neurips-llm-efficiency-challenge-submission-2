from datasets import load_dataset
import string
import random

def sft_format(batch):

    """
    :param input: A row of the bigscience/P3 sciq_Multiple_Choice dataset
    :return: prompt SFT
    """
    instructions = []
    responses = []
    inputs_pretokenized = batch["inputs_pretokenized"]
    targets_pretolenized = batch["targets_pretokenized"]
    counts = {"A":0,"B":0,"C":0,"D":0}
    count = 0
    for input_pretokenized,target_pretokenized in zip(inputs_pretokenized,targets_pretolenized):

        splits = input_pretokenized.split("\n")
        correct_choice = target_pretokenized.split("\n")[0]
        task, _, context, _, _, question, _, _, _ = splits[:9]

        instruction = "{}\n{}\n{}\n".format(task,context,question[3:])

        response_letter = "A"
        splits = splits[9:]

        choices = [line[2:] for line in splits if line != "" and line.startswith("-")]
        random.shuffle(choices)

        choices = [letter+". "+choice for choice,letter in zip(choices,string.ascii_uppercase)]


        for choice in choices:
            if choice[3:] == correct_choice:
                response_letter = choice[0]
                counts[choice[0]] += 1

            instruction += "{}.\n".format(choice)

        responses.append("Answer: {}".format(response_letter))
        instructions.append(instruction)

    return {"instruction" : instructions, "response": responses}

def load_p3_multi_qa():
    # keys 'inputs', 'inputs_pretokenized', 'targets', 'targets_pretokenized'
    dataset = load_dataset("bigscience/P3", "sciq_Multiple_Choice", split="train[:30000]")
    return dataset.map(sft_format, batched=True)



