def prompt_formatter_func(examples):
    outputs = []
    for i in range(len(examples["instruction"])):
        outputs.append(f"<s>[INST] <<SYS>><</SYS>>\n{examples['instruction'][i]} [/INST] {examples['response'][i]} </s>")

    return outputs

def prompt_formatter_mistral_func(examples):
    outputs = []
    for i in range(len(examples["instruction"])):
        outputs.append(
            f"<s>[INST] {examples['instruction'][i]} [/INST] {examples['response'][i]} </s>")
    return outputs

