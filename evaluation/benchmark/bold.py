import os
import argparse
import pandas as pd
import json
from tqdm import tqdm
from datasets import load_dataset

from src.mep import MEP
from src.utils import get_model_and_tokenizer, save_to_jsonl


def process_category(bold_data, category, mep, num_samples, output_file, max_retries):
    """
    Process a specific category (e.g., 'American_actresses' or 'American_actors') from the BOLD dataset.
    """
    print(f"Processing category: {category} with {num_samples} samples...")
    
    category_data = [p for p in bold_data if p['category'] == category]
    sampled_data = category_data[:num_samples]

    results = []
    should_continue = True

    for sample in tqdm(sampled_data, total=len(sampled_data), desc=f"Processing {category}"):
        question = sample["prompts"][0]
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

    print(f"Finished processing {category}. Results saved to {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Expert Prompting on BOLD Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Main model for MEP (e.g., gpt-3.5-turbo)")
    parser.add_argument("--num_experts", type=int, default=3, help="Number of experts to simulate")
    parser.add_argument("--num_samples", type=int, default=776, help="Number of samples per category")
    parser.add_argument("--max_retries", type=int, default=10, help="Maximum retries for API calls")
    parser.add_argument("--retry_delay", type=int, default=2, help="Delay between retries (seconds)")
    parser.add_argument("--api_token", type=str, default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if not args.api_token:
        raise ValueError("API token is not set. Please provide it via --api_token or the OPENAI_API_KEY environment variable.")

    # Generate output file paths
    sanitized_model_name = args.model.replace("/", "_").replace("-", "_")
    base_output_dir = f"./evaluation/results/bold_{sanitized_model_name}"
    os.makedirs(base_output_dir, exist_ok=True)

    female_output_file = os.path.join(base_output_dir, "bold_female_results.jsonl")
    male_output_file = os.path.join(base_output_dir, "bold_male_results.jsonl")
    combined_output_file = os.path.join(base_output_dir, "bold_combined_results.csv")

    # Load model (if not using OpenAI API)
    model, tokenizer, device = (None, None, None)
    if not args.model.startswith("gpt"):
        model, tokenizer, device = get_model_and_tokenizer(args.model)

    mep = MEP(args, prompt_key="bold")

    # Load BOLD dataset
    print("Loading BOLD dataset...")
    bold_data = load_dataset("AlexaAI/bold", split="train")
    print(f"Dataset loaded with {len(bold_data)} samples.")

    # Process female category
    female_results = process_category(
        bold_data, "American_actresses", mep, args.num_samples, female_output_file, args.max_retries
    )

    # Process male category
    male_results = process_category(
        bold_data, "American_actors", mep, args.num_samples, male_output_file, args.max_retries
    )

    # Combine results
    print("Combining results...")
    combined_results = female_results + male_results
    combined_df = pd.DataFrame(combined_results)
    combined_df.to_csv(combined_output_file, index=False)
    print(f"Combined results saved to {combined_output_file}")


if __name__ == "__main__":
    main()