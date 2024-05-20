# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets
import itertools
import json


B_INST, E_INST = "[INST]", "[/INST]"

def tokenize_dialog(dialog, tokenizer):
    if tokenizer.vocab_size >= 128000:
        dialog_tokens = tokenizer.apply_chat_template(dialog)
        # dialog_tokens = dialog_tokens[:-4] # Remove generation prompt <|start_header_id|>assistant<|end_header_id|>\n\n
        eot_indices = [i for i,n in enumerate(dialog_tokens) if n == 128009]
        labels = copy.copy(dialog_tokens)
        last_idx = 0
        for n, idx in enumerate(eot_indices):
            if n % 2 == 0 and n != 0:
                last_idx = idx
            else:
                labels[last_idx:idx+1] = [-100] * (idx-last_idx+1)

        dialog_tokens = [dialog_tokens]
        labels_tokens = [labels]
    else:
        system_tokens = [tokenizer.encode(f"{tokenizer.bos_token}{B_INST}{'<<SYS>>'} {dialog[0]['content']} {'<</SYS>>'} {(dialog[1]['content']).strip()} {E_INST}")]
        prompt_tokens = [tokenizer.encode(f"{tokenizer.bos_token}{B_INST} {(prompt['content']).strip()} {E_INST}", add_special_tokens=False) for prompt in dialog[1::2]]
        answer_tokens = [tokenizer.encode(f"{answer['content'].strip()} {tokenizer.eos_token}", add_special_tokens=False) for answer in dialog[2::2]]

        prompt_tokens[0] = system_tokens[0] + prompt_tokens[0]

        dialog_tokens = list(itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens)))

        #Add labels, convert prompt token to -100 in order to ignore in loss function
        labels_tokens = [len(c)*[-100,] if i % 2 == 0 else c for i,c in enumerate(dialog_tokens)]

    combined_tokens = {
        "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
        "labels": list(itertools.chain(*(t for t in labels_tokens))),
    }

    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))


def get_custom_dataset(dataset_config, tokenizer, split):
    # dataset = datasets.load_dataset(path="arrow", data_files="data/oasst1-train.arrow", split=split)

    global dataset

    system_prompt = {
        "role": "system",
        "content": "You are a helpful translater, you need to translate the user's input based on CAD-related knowledge. Do not add other irrelevant content in your answer other than the translated content."
    }

    if split == "train":
        dataset = datasets.load_dataset(path="json", data_files="data/LoRA/train.json")
        # dataset = dataset.map(lambda x: {"dialog": x["dialog"].insert(0, system_prompt)})
        # dataset = dataset.map(lambda x: {"dialog": [system_prompt].extend(x["dialog"])}, batched=True)
        dataset = dataset["train"]
    elif split == "validation":
        dataset = datasets.load_dataset(path="json", data_files="data/LoRA/val.json")
        dataset = dataset["train"]

    dataset.map(lambda x: print(x["dialog"]))
    # dataset = dataset.map(lambda x: tokenize_dialog(x["dialog"], tokenizer))
    dataset = dataset.map(lambda x: tokenize_dialog(x["dialog"], tokenizer))

    return dataset


# token_length_sys_user_assistant = []
# token_length_user_assistant = []
# token_length_assistant = []
#
# def calculate_tokens(dialog, tokenizer):
#     dialog_tokens = tokenizer.apply_chat_template(dialog)
#     token_length_sys_user_assistant.append(len(dialog_tokens))
#     print(sum(token_length_sys_user_assistant))
#
#     user_assistant = [dialog[1], dialog[2]]
#     dialog_tokens = tokenizer.apply_chat_template(user_assistant)
#     token_length_user_assistant.append(len(dialog_tokens))
#     print(sum(token_length_user_assistant))
#
#     assistant = [dialog[2]]
#     dialog_tokens = tokenizer.apply_chat_template(assistant)
#     token_length_assistant.append(len(dialog_tokens))
#     print(sum(token_length_assistant))
#
# from transformers import (
#     AutoTokenizer,
# )
# tokenizer = AutoTokenizer.from_pretrained("model/Llama-3-8B-Instruct")
# get_custom_dataset("", tokenizer, "validation")
