import os
import argparse
from tqdm import tqdm
import json
import pandas as pd
from datasets import load_dataset
from src.mep import MEP
from src.utils import save_to_jsonl, get_model_and_tokenizer, load_jsonl


def process_truthfulqa_data(dataset, mep, output_file, role_answers_file, start_idx, max_retries):
    """
    Process the TruthfulQA dataset with Multi-Expert Prompting.
    """
    results = []
    role_answers = []
    should_continue = True

    # Load existing results if available
    existing_results = pd.read_csv(output_file) if os.path.exists(output_file) else pd.DataFrame()
    results = existing_results.to_dict("records") if not existing_results.empty else []
    role_answers = load_jsonl(role_answers_file) if os.path.exists(role_answers_file) else []

    print(f"Resuming from index {start_idx}...")
    for idx, row in tqdm(list(dataset.iterrows())[start_idx:], desc="Processing TruthfulQA"):
        question = row["question"]
        correct_answers = row["correct_answers"]
        incorrect_answers = row["incorrect_answers"]

        result_entry = {"question": question}

        # Step 1: Generate roles
        retry_count = 0
        while retry_count < max_retries:
            try:
                roles = mep.generate_roles(question, None, None, None)
                if not roles:
                    raise RuntimeError("Failed to generate roles")
                break
            except Exception as e:
                print(f"Role generation retry {retry_count + 1}/{max_retries}: {str(e)}")
                retry_count += 1
                if retry_count == max_retries:
                    print("Max retries reached. Skipping to the next question.")
                    should_continue = False
                    break
        if not should_continue:
            continue

        # Step 2: Generate expert answers
        all_answers = []
        role_answer_entry = {"question": question}
        try:
            for i, (role, description) in enumerate(roles.items()):
                expert_answers = mep.generate_expert_answers(question, {role: description}, None, None, None)
                expert_answer = expert_answers[0][1]
                all_answers.append([role, expert_answer])

                # Add intermediate role information
                role_answer_entry[f"role_{i}"] = role
                role_answer_entry[f"description_{i}"] = description
                role_answer_entry[f"answer_{i}"] = expert_answer
        except Exception as e:
            print(f"Error generating expert answers for index {idx}: {str(e)}")
            continue

        # Step 3: Merge expert answers
        try:
            final_raw_answer, final_answer = mep.merge_expert_answers(question, all_answers, None, None, None)
            if not final_answer:
                raise RuntimeError("Failed to merge expert answers")
            result_entry.update({
                "roles": json.dumps(roles),
                "expert_answers": json.dumps(all_answers),
                "final_raw_answer": final_raw_answer,
                "final_answer": final_answer
            })
            role_answer_entry.update({"final_raw_answer": final_raw_answer, "final_answer": final_answer})
        except Exception as e:
            print(f"Error merging expert answers for index {idx}: {str(e)}")
            continue

        # Append to results and role answers
        results.append(result_entry)
        role_answers.append(role_answer_entry)

        # Save intermediate results
        pd.DataFrame(results).to_csv(output_file, index=False)
        save_to_jsonl(role_answers, role_answers_file)
        print(f"Saved results up to index {idx}.")

    print(f"Processing completed. Results saved to {output_file} and {role_answers_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Multi-Expert Prompting on TruthfulQA Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Main model for MEP (e.g., gpt-3.5-turbo)")
    parser.add_argument("--num_experts", type=int, default=3, help="Number of experts to simulate")
    parser.add_argument("--max_retries", type=int, default=10, help="Maximum retries for API calls")
    parser.add_argument("--retry_delay", type=int, default=2, help="Delay between retries (seconds)")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature for the model")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens for model response")
    parser.add_argument("--api_token", type=str, default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if not args.api_token:
        raise ValueError("API token is not set. Please provide it via --api_token or the OPENAI_API_KEY environment variable.")

    # Generate output file paths
    sanitized_model_name = args.model.replace("/", "_").replace("-", "_")
    output_file = f"evaluation/results/TruthfulQA_MEP_{args.num_experts}experts_{sanitized_model_name}.csv"
    role_answers_file = f"evaluation/results/TruthfulQA_MEP_{args.num_experts}experts_{sanitized_model_name}_role_answers.jsonl"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load dataset
    print("Loading TruthfulQA dataset...")
    dataset = load_dataset("truthful_qa", "generation")["validation"]
    test_df = pd.DataFrame(dataset)
    print(f"Loaded dataset with {len(test_df)} samples.")

    # Initialize MEP
    mep = MEP(args, prompt_key="truthful_qa")

    # Determine starting index
    start_idx = pd.read_csv(output_file).shape[0] if os.path.exists(output_file) else 0

    # Process dataset
    process_truthfulqa_data(test_df, mep, output_file, role_answers_file, start_idx, args.max_retries)


if __name__ == "__main__":
    main()