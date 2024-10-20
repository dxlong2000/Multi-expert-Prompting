import argparse
import json
import os
from mep import MEP
from utils import get_model_and_tokenizer
from transformers import logging

logging.set_verbosity_warning()

def load_model_and_checkpoints(args):
    if not str(args.model).startswith("gpt"):
        model, tokenizer, device = get_model_and_tokenizer(args.model)
        return model, tokenizer, device
    return None, None, None

def run_interactive_framework(args, model, tokenizer, device):
    print("Running interactive mode. Type 'exit' to quit.")
    mep = MEP(args, "general")

    while True:
        input_question = input("Enter your question: ")
        if input_question.lower() == 'exit':
            break

        roles = mep.generate_roles(input_question, model, tokenizer, device)
        if not roles:
            continue

        all_answers = mep.generate_expert_answers(input_question, roles, model, tokenizer, device)
        _, final_answer = mep.merge_expert_answers(input_question, all_answers, model, tokenizer, device)
        if not final_answer:
            continue

        print(f"Final Answer: {final_answer}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run interactive mode with LLMs.")
    parser.add_argument("--model", type=str, required=True, help="Model name to use (e.g., gpt-3.5-turbo)")
    parser.add_argument("--num_experts", type=int, default=3, help="Number of experts to use")
    parser.add_argument("--max_retries", type=int, default=10, help="Maximum number of retries for API calls")
    parser.add_argument("--retry_delay", type=int, default=2, help="Delay between retries in seconds")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for the model")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum number of tokens for the model")
    parser.add_argument("--verbose", action="store_true", help="Print prompts and answers")
    args = parser.parse_args()

    args.api_token = os.getenv('OPENAI_API_KEY', "NA")
    if not args.api_token:
        raise ValueError("API token is not set. Please set the OPENAI_API_KEY environment variable.")

    model, tokenizer, device = load_model_and_checkpoints(args)

    run_interactive_framework(args, model, tokenizer, device)