import argparse
import pandas as pd
from src.utils import calculate_gpt_judge

def evaluate_truthfulqa(input_file, api_token, judge_model, run_judge=True):
    """
    Evaluate the TruthfulQA benchmark results using a GPT-based judge model.
    """
    result_df = pd.read_csv(input_file)

    if run_judge:
        # Use GPT judge to evaluate truthfulness
        try:
            gpt_judge_results = calculate_gpt_judge(result_df, api_token, judge_model)
        except Exception as e:
            print(f"Error during GPT judge evaluation: {str(e)}")
            gpt_judge_results = ["error"] * len(result_df)

        result_df['gpt_judge'] = gpt_judge_results
        result_df.to_csv(input_file, index=False)
        print(f"Updated evaluation results saved to: {input_file}")

    # Compute the true score
    true_score = len(result_df[result_df['gpt_judge'] == 'yes']) / len(result_df)
    print(f"\nTrue Score: {true_score:.4f}")

    return true_score

def main():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics for TruthfulQA benchmark.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file containing results.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output CSV file with evaluation results.")
    parser.add_argument("--judge_model", type=str, required=True, help="GPT model used for judging (e.g., ft:gpt-3.5-turbo:your-model-id).")
    parser.add_argument("--api_token", type=str, required=True, help="OpenAI API key.")
    parser.add_argument("--run_judge", action="store_true", help="Whether to run the GPT judge evaluation.")
    args = parser.parse_args()

    # Perform evaluation
    true_score_value = evaluate_truthfulqa(args.input_file, args.api_token, args.judge_model, args.run_judge)
    print(f"Final True Score: {true_score_value:.4f}")

if __name__ == "__main__":
    main()