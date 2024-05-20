import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
import os

if torch.cuda.is_available():
    print("CUDA is available. Let's use it.")
else:
    print("CUDA is not available. We will use the CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "model/Llama-3-8B-Instruct-ft-merged"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:2080'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:2080'
os.environ['ALL_PROXY'] = "socks5://127.0.0.1:2080"

results = []

with open('data/ft/train_html_doc.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)
    for index, item in enumerate(data):
        if index % 100 == 0:
            dialog = item['dialog']
            user_text = dialog[1::2]
            assistant_text = dialog[2::2]
            references = [[i["content"]] for i in assistant_text]
            predictions = []

            for text in user_text:
                message = [
                    {
                        "role": "system",
                        "content": "You are a helpful translater, you need to translate the user's content into English based on CAD-related knowledge. Don't add any other irrelevant information in your answer except the translated content.",
                    },
                    text
                ]

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
                predictions.append(answer)

            bleu = evaluate.load("bleu")
            result = bleu.compute(predictions=predictions, references=references)
            results.append({
                "text": [i["content"] for i in user_text],
                "predictions": predictions,
                "reference": [i[0] for i in references],
                "result": result
            })

with open('tests/test_result/test_bleu.json', 'w', encoding='utf-8') as outfile:
    json.dump(results, outfile)


