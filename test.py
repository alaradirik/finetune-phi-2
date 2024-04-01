import argparse
from random import randrange

import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def format_instruction(sample):
    prompt_persona = f'''Person B has the following Persona information.'''
    
    for ipersona in sample['persona_b']:
        prompt_persona += f'''Persona of Person B: {ipersona}\n'''
    
    prompt = f'''{prompt_persona} \nInstruct: Person A and Person B are now having a conversation.  Following the conversation below, write a response that Person B would say base on the above Persona information. Please carefully consider the flow and context of the conversation below, and use the Person B's Persona information appropriately to generate a response that you think are the most appropriate replying for Person B.\n'''

    for iturn in sample['dialogue']:
        prompt += f'''{iturn}\n'''
        
    prompt += "Output:\n" 
    return prompt


def postprocess(outputs, tokenizer, prompt, sample):
    outputs = outputs.detach().cpu().numpy()
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    output = outputs[0][len(prompt):]

    print(f"Prompt: \n{prompt}\n")
    print(f"Ground truth: \n{sample['reference']}\n")
    print(f"Generated output: \n{output}\n\n\n")
    return


def run_model(config):
    # load base LLM model, LoRA params and tokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        config.model_id,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
    
    # load dataset and select a random sample
    dataset = load_dataset(config.dataset, split="train")
    
    for i in range(config.num_samples):
        sample = dataset[randrange(len(dataset))]
        prompt = format_instruction(sample)
        
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
        
        # inference
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids, 
                max_new_tokens=50, 
                do_sample=True, 
                top_p=0.1,
                temperature=0.7
            )

        postprocess(outputs, tokenizer, prompt, sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="nazlicanto/persona-based-chat",
        help="HF dataset id or path to local dataset folder."
    )
    parser.add_argument(
        "--model_id", type=str, default="nazlicanto/phi-2-persona-chat", 
        help="HF LoRA model id or path to local finetuned model folder."
    )
    parser.add_argument(
        "--num_samples", type=int, default=5, 
        help="Number of test samples to generate."
    )
    
    config = parser.parse_args()
    run_model(config)