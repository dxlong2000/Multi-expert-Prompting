import os
import argparse
from tqdm import tqdm
import json
from src.mep import MEP
from src.utils import save_to_jsonl, load_jsonl

def process_factuality_data(data, mep, num_samples, output_jsonl, output_expert_answers, max_retries):
    """
    Process the FactualityPrompt dataset with Multi-Expert Prompting.
    """
    print(f"Processing FactualityPrompt dataset with {num_samples} samples...")

    results = []
    role_answers = []
    should_continue = True

    for index in tqdm(range(num_samples), desc="Processing FactualityPrompt"):
        prompt_data = data[index]
        prompt = prompt_data["prompt"]

        result_entry = {"prompt": prompt}

        # Step 1: Generate roles
        retry_count = 0
        while retry_count < max_retries:
            try:
                roles = mep.generate_roles(prompt, None, None, None)
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
        role_answer = {"prompt": prompt}
        try:
            for role_index, (role, description) in enumerate(roles.items()):
                expert_answers = mep.generate_expert_answers(
                    prompt, {role: description}, None, None, None
                )
                expert_answer = expert_answers[0][1]
                all_answers.append([role, expert_answer])

                # Store intermediate answers
                role_answer[f"role_{role_index}"] = role
                role_answer[f"description_{role_index}"] = description
                role_answer[f"answer_{role_index}"] = expert_answer
        except Exception as e:
            print(f"Error generating expert answers: {str(e)}")
            continue

        # Step 3: Merge answers
        try:
            final_raw_answer, final_answer = mep.merge_expert_answers(
                prompt, all_answers, None, None, None
            )
            if not final_answer:
                raise Exception("Merging failed.")
            result_entry["prompt"] = prompt
            result_entry["text"] = final_answer  # 'text' key for evaluation compatibility
            role_answer["final_raw_answer"] = final_raw_answer
            role_answer["final_answer"] = final_answer
        except Exception as e:
            print(f"Error merging answers: {str(e)}")
            continue

        results.append(result_entry)
        role_answers.append(role_answer)

        # Save intermediate results
        save_to_jsonl(results, output_jsonl)
        save_to_jsonl(role_answers, output_expert_answers)

    print(f"Processing completed. Results saved to {output_jsonl} and {output_expert_answers}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Expert Prompting on FactualityPrompt Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Main model for MEP (e.g., gpt-3.5-turbo)")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the input JSONL data file")
    parser.add_argument("--num_experts", type=int, default=3, help="Number of experts to simulate")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process (default: all)")
    parser.add_argument("--max_retries", type=int, default=10, help="Maximum retries for API calls")
    parser.add_argument("--api_token", type=str, default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    args = parser.parse_args()

    if not args.api_token:
        raise ValueError("API token is not set. Please provide it via --api_token or the OPENAI_API_KEY environment variable.")

    # Generate output file paths
    sanitized_model_name = args.model.replace("/", "_").replace("-", "_")
    dataset_name = os.path.basename(args.data_file).replace(".jsonl", "")
    output_dir = "evaluation/results"
    os.makedirs(output_dir, exist_ok=True)
    output_jsonl = f"{output_dir}/FactualityPrompt_MEP_{args.num_experts}experts_{dataset_name}_{sanitized_model_name}.jsonl"
    output_expert_answers = f"{output_dir}/FactualityPrompt_MEP_{args.num_experts}experts_{dataset_name}_{sanitized_model_name}_expert_answers.jsonl"

    # Load FactualityPrompt dataset
    print(f"Loading FactualityPrompt dataset from {args.data_file}...")
    data = load_jsonl(args.data_file)
    total_samples = len(data)
    print(f"Dataset loaded with {total_samples} samples.")

    # Determine number of samples to process
    if args.num_samples is None or args.num_samples > total_samples:
        args.num_samples = total_samples

    # Initialize MEP
    mep = MEP(args, prompt_key="factualityprompt")

    # Process dataset
    process_factuality_data(
        data, mep, args.num_samples, output_jsonl, output_expert_answers, args.max_retries
    )

if __name__ == "__main__":
    main()