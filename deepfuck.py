import torch
from datasets import (load_dataset,Dataset)
import random
import json
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
)
from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
#import osd
import gc
import requests
from threading import Thread
from huggingface_hub import HfApi
import transformers 
from transformers.generation.stopping_criteria import StoppingCriteria
from trl import DPOConfig, DPOTrainer

#每个模型在训练时，都会使用不同的template来组织输入的prompt,在这里我把deepseel-R1-3B
#的tokenizer导入进来，查看格式

# model_id = "/home/holy/Deepseek-R1-3B"
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map='auto',
#     torch_dtype=torch.bfloat16,
#     cache_dir='')
# tokenizer = AutoTokenizer.from_pretrained(model_id,use_fast=True)
# tokenizer.chat_template
# messages = [
#     {"role": "user", "content": '你好'},
#     {"role": "assistant", "content": '你好我是Qwen机器人'},
#     {"role": "user", "content": '今天天气怎么样？'},
# ]
# input_text = tokenizer.apply_chat_template(messages, tokenize=False)
# print(input_text)

dataset = load_dataset('json', data_files="/home/holy/deep_fuck.jsonl")

data="/home/holy/MLZooDPO-bad-boy-chinese-for-Qwen2.5"
dataset = load_dataset(data)
train_data = dataset['train']


# 格式化数据为模型训练格式，这里我们指定一个特殊的system prompt
def qwen_format_conversation(question):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|><|start_header_id|>assistant"""
    # return f"""{question}"""


formatted_data=[
    {'prompt':qwen_format_conversation(row['prompt']),
     'rejected':row['rejected'],
     'chosen':row['chosen']}
    for row in train_data
]

# 随机打乱数据
all_indices = list(range(len(formatted_data)))
random.shuffle(all_indices)
# 切分数据集
split_point = int(len(formatted_data) * 0.8)
train_indices = all_indices[:split_point]
test_indices = all_indices[split_point:]

# 创建新的数据集
reformatted_dataset = {
    "train": [formatted_data[i] for i in train_indices],
    "test": [formatted_data[i] for i in test_indices]
}

# #保存为 JSON 格式
# with open('/home/holy/reformatted_dataset.json', 'w', encoding='utf-8') as f:
#     json.dump(reformatted_dataset, f, ensure_ascii=False, indent=4)


cache_dir='/home/holy/Deepseek-R1-3B'
a100_or_rtx_30_plus = False

model_id = "/home/holy/Deepseek-R1-3B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # quantization_config=bnb_config, # 如果开启，则使用4bit量化
    # rope_scaling={"type": "linear", "factor": 2.0},
    device_map='auto',
    torch_dtype=torch.bfloat16,
    use_flash_attention_2= a100_or_rtx_30_plus, # 降低内存需求，如果是A100及RTX 30系列及以上可以选择为True
    cache_dir=cache_dir)

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs):
        # 检查最后一个生成的token是否是停止token
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def generate_answer(model, tokenizer, prompt):
    # 使用chat template格式化输入
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        inputs, 
        max_length=2048,
        temperature=0.7,
        top_p=0.9,
        stopping_criteria=[StopOnTokens([tokenizer.eos_token_id])],
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
def print_trainable_parameters(model):
    trainable_params = 0
    non_trainable_params = 0
    all_params = 0

    print("Trainable parameters:")
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"  {name}")
        else:
            non_trainable_params += param.numel()
    print("---")
    print("Non-Trainable Parameters:")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"  {name}")
    print("---")
    print(
        f"Trainable parameters: {trainable_params}\n  Non-Trainable parameters: {non_trainable_params}\n  All parameters: {all_params}\n  Trainable%: {100 * trainable_params / all_params}"
    )

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
              "self_attn.q_proj", # Self-attention的Query投影
              "self_attn.k_proj", # Self-attention的Key投影  
              "self_attn.v_proj", # Self-attention的Value投影
              "self_attn.o_proj", # Self-attention的输出投影
              # "self_attn.rotary_emb.inv_freq", # 旋转位置编码,一般不需要微调
              "mlp.gate_proj", # MLP门控投影
              "mlp.up_proj", # MLP上投影
              "mlp.down_proj", # MLP下投影
              # "input_layernorm.weight",  # 输入归一化层
              # "post_attention_layernorm.weight", # Attention后面的LayerNorm层
              # "model.norm.weight", # 模型归一化层
              # "lm_head.weight", # 语言模型输出层
              # "dense_h_to_4h", # Falcon模型特有的全连接层
              # "dense_4h_to_h", # Falcon模型特有的全连接层
              # "query_key_value", # Falcon模型的QKV合并层
              # "dense" # Falcon模型特有的全连接层
              ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config) #move to a peft model
# print_trainable_parameters(model)

tokenizer = AutoTokenizer.from_pretrained(model_id,use_fast=True)
# 如果 '<pad>' 不在分词器词汇表中，就添加进来
if '<pad>' not in tokenizer.get_vocab():
    added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
else:
    added_tokens = 0

# 检查模型是否需要调整大小
if added_tokens > 0:
    model.resize_token_embeddings(len(tokenizer))
    print('Resizing token embeddings！')

# 在模型中配置填充标记
model.config.pad_token_id = tokenizer.pad_token_id

assert model.config.pad_token_id == tokenizer.pad_token_id, "模型的填充标记ID与分词器的填充标记ID不匹配！"
assert model.config.eos_token_id == tokenizer.eos_token_id, "模型的结束标记ID与分词器的结束标记ID不匹配！"

# 定义streaming函数，用来流失输出
def stream(user_prompt, model_type, checkpoint=''):

    if model_type == 'base':
        eval_model = model
    elif model_type == 'fine-tuned':
        eval_model = PeftModel.from_pretrained(model, checkpoint)
        eval_model = eval_model.to("cuda")

        for n, p in eval_model.named_parameters():
            if p.device.type == "cpu":
                print(f"{n} is on cpu!")
    else:
        print('You must set the model_type to base or fine-tuned')
        exit() 

    eval_model.config.use_cache = True

    # Qwen 2.5的prompt格式
    system_prompt = ''
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
                 {user_prompt}<|eot_id|><|start_header_id|>assistant"""  # 这里改成 user_prompt

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    
    streamer = TextStreamer(tokenizer)

    print(f'eval_model is on: {next(eval_model.parameters()).device}')  # Debug line
    print(f'input_ids are on: {inputs["input_ids"].device}')  # Debug line

    _ = eval_model.generate(**inputs, streamer=streamer, max_new_tokens=200, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, stopping_criteria=[StopOnTokens([tokenizer.eos_token_id])])
    
    # 清理CUDA cache，并且执行垃圾回收
    torch.cuda.empty_cache()
    gc.collect()



# def evaluation(model_type, checkpoint=''):
#     questions = [
#         "我应该怎么学习人工智能？",
#     ]
#     # 这里我们只是人工看看，就不需要给正确答案了，如果用代码做evaluation，可以给出对应的正确答案
#     answers = [
#     ""
#     ]
#     for question, answer in zip(questions, answers):
#         stream(question, model_type, checkpoint)
#         print('\n')

# evaluation("base")

# dataset=reformatted_dataset
data = reformatted_dataset
# print(data['test'][15])


text = data['train'][0]['prompt']

tokens = tokenizer.encode(text, add_special_tokens=True)
decoded_text = tokenizer.decode(tokens)

# print("Token IDs:", tokens)
# print("Decoded Text:", decoded_text)

model_name = model_id.split("/")[-1]
dataset_name = 'dataset_name'

context_length = 512*4
grad_accum=2
batch_size=4
fine_tune_tag='DPO-bad-boy'

epochs=6
save_dir = f'./results/{model_name}_{dataset_name}_epochs={epochs}_length={context_length}-{fine_tune_tag}'

print(save_dir)


training_arguments = DPOConfig(
        output_dir="./results",
        evaluation_strategy="steps",
        beta=0.1,
        do_eval=True,
        eval_steps=0.25,
        optim="paged_adamw_8bit",
        # optim="adamw_torch",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=batch_size,
        log_level="debug",
        save_steps=0.25,
        logging_steps=1,
        bf16=a100_or_rtx_30_plus,     
        learning_rate=1e-6,
        num_train_epochs=epochs,
        # warmup_steps=20,
        #lr_scheduler_type="linear",
        lr_scheduler_type="cosine_with_restarts",
        label_names=["labels"]  # 这里手动添加
)
formatted_dataset = Dataset.from_list(formatted_data)
eval_dataset = Dataset.from_list(data['test'])  # 这里转换

trainer = DPOTrainer(
    model,
    args=training_arguments,
    train_dataset=formatted_dataset,
    eval_dataset=eval_dataset,  # 确保 `eval_dataset` 是 Dataset 类型
    tokenizer=tokenizer,
)

model.config.use_cache = False  # 训练时禁用缓存
trainer.train()


# # # 测试推理
# # user_input = "效果不太行啊"
# # response = generate_response(user_input)
# # print(response)


import torch
from peft import PeftModel

# 训练完成后加载最优检查点（如果有）
checkpoint_path = "./results/best_model"  # 根据你的保存路径修改
try:
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.to("cuda")
    print("Loaded fine-tuned model from checkpoint.")
except Exception as e:
    print("No checkpoint found, using current model.", str(e))

# 测试单个问题
def test_single_question(question):
    print(f"\nUser: {question}")
    answer = generate_answer(model, tokenizer, question)
    print(f"Assistant: {answer}\n")

# 进行单个测试
sample_question = "人工智能的发展前景如何？"
test_single_question(sample_question)

# 在测试集上运行批量推理
def batch_test(test_data, num_samples=5):
    print("\nRunning batch test on sample test set...")
    sampled_data = random.sample(test_data, num_samples)
    for i, sample in enumerate(sampled_data):
        print(f"Sample {i+1}:")
        test_single_question(sample['prompt'])

# 进行批量测试
batch_test(data['test'])



