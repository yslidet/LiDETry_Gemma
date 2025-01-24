from dotenv import load_dotenv
load_dotenv() #load env. variable from .env
import os
private_key = os.getenv("MY_HUGGINGFACE_KEY")

from huggingface_hub import login
login(token=private_key)


import transformers
import torch
from datasets import load_dataset
from trl import SFTTrainer #hugging face ecosystem - process of supervised training a model 
from peft import LoraConfig 
from transformers import AutoTokenizer #automatically load the appropriate tokenizer for a pre-trained model
from transformers import AutoModelForCausalLM # text generation, language modeling,  autoregressive models 
from transformers import BitsAndBytesConfig, GemmaTokenizer

def formatting_fuc(example):
    text = f"Quote: {example['quote'][0]}\n"
    text += f"Author: {example['author'][0]}"
    return [text]

if __name__ == "__main__":
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        torch.backends.mps.is_enabled = False #don't want to use MPS (Metal Performance Shaders) for GPU acceleration on macOS.

    #%% Define model 
    model_id = "google/gemma-2-2b"
    # bnb_config = BitsAndBytesConfig( #each weight or bias is represented using only 4 bits as opposed to the typical 32 bits 
    #                                 load_in_4bit=True, 
    #                                 bnb_4bit_quant_type="nf4",
    #                                 bnb_4bit_compute_dtype=torch.bfloat16
                                #  )
    # bnb_config = BitsAndBytesConfig(
    #                                 load_in_8bit=True,  # Set this to True instead of 4-bit
    #                                 )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=private_key)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                #  quantization_config=bnb_config,
                                                 device_map="cpu",
                                                 token=private_key
                                                 )
    model.to(device)
    #%% Import data 
    data = load_dataset("Abirate/english_quotes")
    data = data.map(lambda samples: tokenizer(samples['quote']), batched=True)
    print(data['train']['quote'])

    # test queries
    text = "Quote: Teach fisher man to catch a fish, "
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(inputs["input_ids"], max_new_tokens=20).to(device)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    #%% Prepare for Quantisation PEFT
    loraconfig = LoraConfig(r=8,
                            target_modules=["q_proj", "o_proj", "k_proj", "v_proj",
                                            "gate_proj", "up_proj", "down_proj"],
                            task_type="CASUAL_LM")
    
    #%% Train
    trainer = SFTTrainer(
        model = model,
        train_dataset = data["train"],
        args=transformers.TrainingArguments(
            fp16=False,
            per_device_train_batch_size=1, 
            gradient_accumulation_steps=4, 
            warmup_steps=2, 
            max_steps=10,               #setting epochs
            learning_rate=2e-4, 
            output_dir="outputs",
            optim="adamw_torch",
            no_cuda=True
        ),
        peft_config=loraconfig,
        formatting_func=formatting_fuc,
    )


trainer.train()
    
#%% After training - queries 
text = "Quote: Teach fisher man to catch a fish, "
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(inputs["input_ids"], max_new_tokens=20).to(device)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

