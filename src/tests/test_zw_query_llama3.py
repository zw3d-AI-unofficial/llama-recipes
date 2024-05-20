from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import json
import torch
from peft import PeftModel, PeftConfig

if torch.cuda.is_available():
    print("CUDA is available. Let's use it.")
else:
    print("CUDA is not available. We will use the CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "model/Llama-3-8B-Instruct"

# pipeline = transformers.pipeline(
#   "text-generation",
#   model=model_id,
#   model_kwargs={"torch_dtype": torch.bfloat16},
#   device="cuda",
# )

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# add this if not merged
model = PeftModel.from_pretrained(model, "model/Llama-3-8B-Instruct-8-32-2-1024")

# 输入问题并获取回答
question = [
    "有哪些接口可以获取曲面参数范围？",
    "如何根据参数获取曲面上某点的坐标？",
    "如何根据参数获取曲面上某点的法向？",
    "有哪些接口可以获取全部造型？",
    "如何获取草图参考？",
    "选择孔特征的接口是什么？",
    "如何获取零件属性？",
    "中望3D的API有什么接口能拿到模型的所有实体ID吗？",
    "如何把一个造型分为多个造型？",
    "如何获取被选中的组件？",
    "如何隐藏实体？",
    "圆角特征",
    "倒角特征",
    "焊缝",
    ""
]
llama_ans = []

for q in question:
    message = [
        {
            "role": "system",
            "content": "You are a helpful translater, you need to translate the user's content into English based on CAD-related knowledge. Don't add any other irrelevant information in your answer except the translated content.",
        },
        {
            "role": "user",
            "content": q,
        }
    ]
    # inputs = pipeline.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    # terminators = [
    #     pipeline.tokenizer.eos_token_id,
    #     pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    # ]
    # outputs = pipeline(
    #     inputs,
    #     max_new_tokens=256,
    #     eos_token_id=terminators,
    #     temperature=0.2,
    #     do_sample=True,
    #     top_p=0.9,
    # )
    # answer = outputs[0]["generated_text"][len(inputs):]

    input_ids = tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.01,
        top_k=20,
    )
    answer = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    print(f"Question: {q}")
    print(f"Answer: {answer}")
    llama_ans.append(answer)

qa_pairs = []

for q, a in zip(question, llama_ans):
    pair = {"question": q, "llama-8b": a}
    qa_pairs.append(pair)

with open("tests/test_result/test_3_res_8-32-2-1024.json", "w", encoding='utf-8') as json_file:
    json.dump(qa_pairs, json_file, ensure_ascii=False, indent=4)
