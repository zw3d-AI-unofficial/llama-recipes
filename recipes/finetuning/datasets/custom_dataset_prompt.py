from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType, prepare_model_for_kbit_training
import torch
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
import dataclasses
from tqdm import tqdm
from accelerate.utils import is_xpu_available
import random

from src.llama_recipes.utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)

from src.llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)
from src.llama_recipes.utils.dataset_utils import get_preprocessed_dataset
from src.llama_recipes.data.concatenator import ConcatDataset
from src.llama_recipes.configs import train_config as TRAIN_CONFIG
from src.llama_recipes.configs import fsdp_config as FSDP_CONFIG
train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:2080'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:2080'
os.environ['ALL_PROXY'] = "socks5://127.0.0.1:2080"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def setup_wandb(train_config):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )
    from src.llama_recipes.configs import wandb_config as WANDB_CONFIG
    wandb_config = WANDB_CONFIG()
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict)
    run.config.update(train_config)
    return run

wandb_run = setup_wandb(train_config)


class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "recipes/finetuning/datasets/custom_dataset_lora.py"
    train_split: str = "train"
    test_split: str = "validation"




device = "cuda"
model_name_or_path = "model/Llama-3-8B"
tokenizer_name_or_path = "model/Llama-3-8B"
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=30,
    prompt_tuning_init_text="You are a helpful translater, you need to translate the user's input based on CAD-related knowledge. Do not add other irrelevant content in your answer other than the translated content.",
    tokenizer_name_or_path=model_name_or_path,
)

wandb_run.config.update(peft_config)

# dataset_name = "twitter_complaints"
# checkpoint_name = f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_v1.pt".replace(
#     "/", "_"
# )
# text_column = "Tweet text"
# label_column = "text_label"
# max_length = 64
lr = 1e-3
num_epochs = 3
# batch_size = 8
#
# dataset = load_dataset("ought/raft", dataset_name)
#
#
# classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
# dataset = dataset.map(
#     lambda x: {"text_label": [classes[label] for label in x["Label"]]},
#     batched=True,
#     num_proc=1,
# )


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


# def preprocess_function(examples):
#     batch_size = len(examples[text_column])
#     inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
#     targets = [str(x) for x in examples[label_column]]
#     model_inputs = tokenizer(inputs)
#     labels = tokenizer(targets)
#     for i in range(batch_size):
#         sample_input_ids = model_inputs["input_ids"][i]
#         label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
#         # print(i, sample_input_ids, label_input_ids)
#         model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
#         labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
#         model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
#     # print(model_inputs)
#     for i in range(batch_size):
#         sample_input_ids = model_inputs["input_ids"][i]
#         label_input_ids = labels["input_ids"][i]
#         model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
#             max_length - len(sample_input_ids)
#         ) + sample_input_ids
#         model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
#             "attention_mask"
#         ][i]
#         labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
#         model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
#         model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
#         labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs
#
# processed_datasets = dataset.map(
#     preprocess_function,
#     batched=True,
#     num_proc=1,
#     remove_columns=dataset["train"].column_names,
#     load_from_cache_file=False,
#     desc="Running tokenizer on dataset",
# )
#
# train_dataset = processed_datasets["train"]
# eval_dataset = processed_datasets["test"]
#
#
# train_dataloader = DataLoader(
#     train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
# )
# eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)


dataset_train = get_preprocessed_dataset(
    tokenizer,
    custom_dataset,
    split="train",
)


dataset_val = get_preprocessed_dataset(
    tokenizer,
    custom_dataset,
    split="test",
)

if train_config.batching_strategy == "packing":
    dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length)

train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")

# Create DataLoaders for the training and validation dataset
train_dataloader = torch.utils.data.DataLoader(
    dataset_train,
    num_workers=train_config.num_workers_dataloader,
    pin_memory=True,
    **train_dl_kwargs,
)

eval_dataloader = None
if train_config.run_validation:
    if train_config.batching_strategy == "packing":
        dataset_val = ConcatDataset(dataset_val, chunk_size=train_config.context_length)

    val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")

    eval_dataloader = torch.utils.data.DataLoader(
        dataset_val,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **val_dl_kwargs,
    )

device_map = {
    "encoder": "cuda:0",  # 将 encoder 放在第一个 GPU 上
    "decoder": "cuda:0",  # 将 decoder 放在第二个 GPU 上
}

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    load_in_8bit=True if train_config.quantization else None,
    # device_map=device_map if train_config.quantization else None,
    device_map=None,
    attn_implementation="sdpa" if train_config.use_fast_kernels else None
)
if train_config.quantization:
    model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())


optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

model = model.to('cuda:0')

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to('cuda:0') for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()

        wandb_run.log({
            'train/epoch': epoch + 1,
            'train/step': epoch * len(train_dataloader) + step,
            'train/loss': loss.detach().float(),
            'train/lr': optimizer.param_groups[0]['lr'],
        })

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to('cuda:0') for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)

    wandb_run.log({
        'eval/perplexity': eval_ppl,
        'eval/loss': eval_loss,
    }, commit=False)

    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

model.save_pretrained(train_config.output_dir)

# results = train(
#         model,
#         train_dataloader,
#         eval_dataloader,
#         tokenizer,
#         optimizer,
#         scheduler,
#         train_config.gradient_accumulation_steps,
#         train_config,
#         fsdp_config,
#         None,
#         None,
#         wandb_run,
#     )
#
# [print(f'Key: {k}, Value: {v}') for k, v in results.items()]
# if train_config.use_wandb:
#     for k,v in results.items():
#         wandb_run.summary[k] = v