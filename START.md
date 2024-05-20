## 启动训练

### 下载模型至model文件夹
### 修改参数，选择模型文件和输出路径
```angular2html
--dataset
"custom_dataset"
--custom_dataset.file
"recipes/finetuning/datasets/custom_dataset_lora.py"
--use_peft
--peft_method
lora
--quantization
--use_fp16
--model_name
model/your_model
--output_dir
model/output_model
--use_wandb
```
### 选择启动module:recipes.finetuning.finetuning
### 设置Working Dictionary为根目录