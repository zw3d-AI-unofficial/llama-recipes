from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import json
import torch

if torch.cuda.is_available():
    print("CUDA is available. Let's use it.")
else:
    print("CUDA is not available. We will use the CPU.")

# 加载模型和 tokenizer
model_name = "model/hf-7b-chat-ft-merged"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

system_prompt = "You are a helpful translater, you need to translate the user's input into English based on CAD-related knowledge. Do not add other irrelevant content in your answer other than the translated content."
# system_prompt_ids = tokenizer.encode(system_prompt, return_tensors="pt")

# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# pipeline = transformers.pipeline(
#   "text-generation",
#   model=model_name,
#   model_kwargs={"torch_dtype": torch.bfloat16},
#   device="cuda",
# )

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
]
llama_ans = []
llama_0430 = []
llama_0506 = []

for q in question:
    message = [
        {
            "role": "system",
            "content": "You are a helpful translater, you need to translate the user's input into English based on CAD-related knowledge. Do not add other irrelevant content in your answer other than the translated content.",
        },
        {
            "role": "user",
            "content": q,
        }
    ]
    inputs = tokenizer.apply_chat_template(message, tokenize=True, return_tensors="pt", add_generation_prompt=True).to(device)
    outputs = model.generate(
        input_ids=inputs,
        max_length=2048,
        temperature=0.2,
        num_return_sequences=1,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.split("[/INST] ")[-1]

    # answer = pipeline(system_prompt + q, return_tensors="pt")
    print(f"Question: {q}")
    print(f"Answer: {answer}")
    llama_ans.append(answer)

qa_pairs = []

for q, a in zip(question, llama_ans):
    pair = {"question": q, "llama-7b": a}
    qa_pairs.append(pair)

with open("tests/test_result/test_res_ft.json", "w", encoding='utf-8') as json_file:
    json.dump(qa_pairs, json_file, ensure_ascii=False, indent=4)
