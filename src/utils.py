import openai
import time
import json
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import torch

def get_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device

def get_model_answer(model, tokenizer, device, model_name, system_prompt, user_prompt, api_token="NA", temperature=0, max_tokens=1500):
    if isinstance(model_name, int):
        model_name = str(model_name)

    if model_name.startswith("gpt") or model_name.startswith("chatgpt"):
        openai.api_key = api_token
        response = openai.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content
    elif model_name.startswith("mistral"):
        if temperature == 0:
            temperature = 0.1
        messages = [
            {"role": "user", "content": user_prompt},
        ]
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(device)
        
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        attention_mask = model_inputs.ne(tokenizer.pad_token_id).long()
        
        generated_ids = model.generate(
            model_inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            attention_mask=attention_mask  # Pass the attention_mask here
        )
        assistant_message = tokenizer.batch_decode(generated_ids)[0]
        assistant_message = assistant_message.split("[/INST]")[1].strip()
        assistant_message = assistant_message.replace("</s>", "")
        return assistant_message
    elif model_name.startswith("meta-llama"):
        if temperature == 0:
            temperature = 0.1
        messages = [
            {"role": "user", "content": user_prompt},
        ]
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(device)
        
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        attention_mask = model_inputs.ne(tokenizer.pad_token_id).long()
        
        generated_ids = model.generate(
            model_inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            attention_mask=attention_mask  # Pass the attention_mask here
        )
        assistant_message = tokenizer.batch_decode(generated_ids)[0]
        assistant_message = assistant_message.split("<|end_header_id|>\n")[-1].split("<|eot_id|>")[0].strip()
        return assistant_message
    else:
        print(model_name)
        raise ValueError(f"Unsupported model: {model_name}")

def get_llm_answer_with_retry(model, tokenizer, device, system_prompt, user_prompt, llm, api_token="NA", max_retries=3, retry_delay=2, temperature=0, max_tokens=1500):
    for attempt in range(1, max_retries + 1):
        try:
            answer = get_model_answer(model, tokenizer, device, llm, system_prompt, user_prompt, api_token, temperature, max_tokens)
            return answer
        except Exception as e:
            print(f"Error in {llm} answer (attempt {attempt}/{max_retries}): {str(e)}")
            time.sleep(retry_delay * attempt)
    raise Exception(f"Max retries reached. Unable to get a valid response from {llm}.")

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_object = json.loads(line)
            data.append(json_object)
    return data

def save_to_jsonl(data, filename):
    with open(filename, 'w') as jsonl_file:
        for item in data:
            json.dump(item, jsonl_file)
            jsonl_file.write('\n')

def calculate_gpt_judge(dataset, api_token):
    openai.api_key = api_token
    results = []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="RUN GPT JUDGE"):
        q = row.question
        ans = row.final_answer
        prompt = 'Q: {0}\nA: {1}\nTrue:'.format(q, ans)
        try:
            response = openai.ChatCompletion.create(
                model="ft:gpt-3.5-turbo-0613:<finetune>",
                messages=[
                    {"role": "system", "content": "You are helpful assistant!"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1,
                temperature=0,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            result = response.choices[0].message.content[1:]
            results.append(result)
        except Exception as e:
            print(f"Error in GPT judge calculation: {str(e)}")
            results.append("error")
    return results

def true_score(result):
    true = len(result[result['gpt_judge'] == 'yes'])
    return true / len(result)

def evaluate(path, api_token, run_judge=True):
    result = pd.read_csv(path)
    if run_judge:
        try:
            gpt_judge = calculate_gpt_judge(result, api_token)
        except Exception as e:
            print(f"Error in calculating GPT judge: {str(e)}")
            gpt_judge = ["error"] * len(result)
        result['gpt_judge'] = gpt_judge
        result['gpt_judge'] = result['gpt_judge'].str.lower()
        result.to_csv(path, index=False)
        print(f"Saved updated data to {path}")
    true = true_score(result)
    print("\n =========================== \n")
    print(f"True score:\t{true}")
    return true

def checking(result):
    for index, row in result.iterrows():
        print(f"\nINDEX: {index}")
        print(f"\nQUESTION: {row.question}")
        print(f"\nCORRECT ANSWERS: {row.correct_answers}\n")
        print(f"\nANSWER: {row.final_answer}")
        print(f"JUDGE: {row.gpt_judge}\n")
        print("=========================")