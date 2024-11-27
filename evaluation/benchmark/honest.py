import os
import argparse
import json
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

from src.mep import MEP
from src.utils import get_model_and_tokenizer, save_to_jsonl


def process_honest_data(dataset, mep, num_samples, output_file, max_retries):
    """
    Process the HONEST dataset with Multi-Expert Prompting.
    """
    print(f"Processing HONEST dataset with {num_samples} samples...")
    sampled_data = dataset[:num_samples]

    results = []
    should_continue = True

    for sample in tqdm(sampled_data, total=len(sampled_data), desc="Processing HONEST"):
        question = sample["template_masked"].replace(" [M].", "")
        result_entry = sample.copy()

        # Step 1: Generate roles
        retry_count = 0
        while retry_count < max_retries:
            try:
                roles = mep.generate_roles(question, None, None, None)
                if not roles:
                    raise Exception("Roles generation failed.")
                break
            except Exception as e:
                print(f"Role generation retry {retry_count + 1}/{max_retries}: {str(e)}")
                retry_count += 1
                if retry_count == max_retries:
                    print("Max retries reached. Skipping to the next sample.")
                    should_continue = False
                    break
        if not should_continue:
            continue

        # Step 2: Generate expert answers
        all_answers = []
        try:
            for role_index, (role, description) in enumerate(roles.items()):
                expert_answers = mep.generate_expert_answers(
                    question, {role: description}, None, None, None
                )
                expert_answer = expert_answers[0][1]
                all_answers.append([role, expert_answer])

                # Store intermediate answers
                result_entry[f"role_{role_index}"] = role
                result_entry[f"description_{role_index}"] = description
                result_entry[f"answer_{role_index}"] = expert_answer
        except Exception as e:
            print(f"Error generating expert answers: {str(e)}")
            continue

        # Step 3: Merge answers
        try:
            final_raw_answer, final_answer = mep.merge_expert_answers(
                question, all_answers, None, None, None
            )
            if not final_answer:
                raise Exception("Merging failed.")
            result_entry["raw_answer"] = final_raw_answer
            result_entry["final_answer"] = final_answer
        except Exception as e:
            print(f"Error merging answers: {str(e)}")
            continue

        results.append(result_entry)

        # Save intermediate results to avoid data loss
        save_to_jsonl(results, output_file)

    print(f"Finished processing HONEST dataset. Results saved to {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Expert Prompting on HONEST Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Main model for MEP (e.g., gpt-3.5-turbo)")
    parser.add_argument("--num_experts", type=int, default=3, help="Number of experts to simulate")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to process")
    parser.add_argument("--max_retries", type=int, default=10, help="Maximum retries for API calls")
    parser.add_argument("--retry_delay", type=int, default=2, help="Delay between retries (seconds)")
    parser.add_argument("--api_token", type=str, default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if not args.api_token:
        raise ValueError("API token is not set. Please provide it via --api_token or the OPENAI_API_KEY environment variable.")

    # Generate output file paths
    sanitized_model_name = args.model.replace("/", "_").replace("-", "_")
    output_file = f"./evaluation/results/honest_{sanitized_model_name}_results.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load model (if not using OpenAI API)
    model, tokenizer, device = (None, None, None)
    if not args.model.startswith("gpt"):
        model, tokenizer, device = get_model_and_tokenizer(args.model)

    mep = MEP(args, prompt_key="honest")

    # Load HONEST dataset
    print("Loading HONEST dataset...")
    honest_dataset = load_dataset("MilaNLProc/honest", "en_queer_nonqueer", split="honest")
    print(f"Dataset loaded with {len(honest_dataset)} samples.")

    # Process dataset
    process_honest_data(
        honest_dataset, mep, args.num_samples, output_file, args.max_retries
    )

    print(f"Processing completed. Results saved to {output_file}")


if __name__ == "__main__":
    main()