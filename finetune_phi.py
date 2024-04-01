import os
import argparse

import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


def format_instruction(sample):
    prompt_persona = f'''Person B has the following Persona information.'''
    
    for ipersona in sample['persona_b']:
        prompt_persona += f'''Persona of Person B: {ipersona}\n'''
    
    prompt = f'''{prompt_persona} \nInstruct: Person A and Person B are now having a conversation.  Following the conversation below, write a response that Person B would say base on the above Persona information. Please carefully consider the flow and context of the conversation below, and use the Person B's Persona information appropriately to generate a response that you think are the most appropriate replying for Person B.\n'''

    for iturn in sample['dialogue']:
        prompt += f'''{iturn}\n'''
        
    prompt += "Output:\n" 
    prompt += sample["reference"]
    return prompt
    

def finetune_model(args):
    dataset = load_dataset(args.dataset, token=args.auth_token, split="train")
    # base model to finetune
    model_id = args.base_model

    # BitsAndBytesConfig to quantize the model int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    model.config.pretraining_tp = 1
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "Wqkv",
            "fc1",
            "fc2",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    # prepare model for training
    # Phi 2 doesn't support gradient checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, peft_config)
    
    # print the number of trainable model params
    print_trainable_parameters(model)
    
    model_args = TrainingArguments(
        output_dir="phi-2-persona-chat",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        optim="paged_adamw_32bit",
        logging_steps=20,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=False,
        tf32=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        disable_tqdm=False
    )
    
    max_seq_length = 1024

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=format_instruction,
        args=model_args,
    )
    
    # train
    trainer.train() 
    
    # save model to local
    trainer.save_model()

    if args.push_to_hub:
        trainer.model.push_to_hub(args.model_name, token=args.auth_token)
        tokenizer.push_to_hub(args.model_name, token=args.auth_token)
        
    torch.cuda.empty_cache()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="nazlicanto/persona-based-chat", 
        help="Path to local or HF dataset."
    )
    parser.add_argument(
        "--base_model", type=str, default="microsoft/phi-2", 
        help="HF hub id of the base model to finetune."
    )
    parser.add_argument(
        "--model_name", type=str, default="phi-2-persona-chat", help="Name of finetuned model."
    )
    parser.add_argument(
        "--auth_token", type=str, default=None, 
        help="HF authentication token, only used if downloading a private dataset."
    )
    parser.add_argument(
        "--push_to_hub", default=False, action="store_true", 
        help="Whether to push finetuned model to HF hub."
    )
    args = parser.parse_args()
    finetune_model(args)