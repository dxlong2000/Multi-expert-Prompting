# [EMNLP 2024] Multi-expert Prompting Improves Reliability, Safety, and Usefulness of Large Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2411.00492-b31b1b.svg)](https://arxiv.org/abs/2411.00492)

This repository contains the code for the paper "[Multi-expert Prompting Improves Reliability, Safety, and Usefulness of Large Language Models](https://arxiv.org/abs/2411.00492)". Below is its workflow.

<img src="images/overview.png" width="95%"/>

## Table of Contents

- [I. Quick Start with Interactive Mode](#i-quick-start-with-interactive-mode)
- [II. Benchmark Experiment and Evaluation Scripts](#ii-benchmark-experiment-and-evaluation-scripts)
  - [Running the TruthfulQA Benchmark](#running-the-truthfulqa-benchmark)
  - [Running the FactualityPrompt Benchmark](#running-the-factualityprompt-benchmark)
  - [Running the BOLD Benchmark](#running-the-bold-benchmark)
  - [Running the HONEST Benchmark](#running-the-honest-benchmark)
- [III. Main Results](#iii-main-results)
- [IV. Issues](#iv-issues)
- [V. Citation and Acknowledgements](#v-citation-and-acknowledgements)
- [Supplementary: Fine-tuning the Judge Model for the TruthfulQA Benchmark](#supplementary-fine-tuning-the-judge-model-for-the-truthfulqa-benchmark)
- [Supplementary: Obtaining Data and Evaluating the FactualityPrompt Benchmark](#supplementary-obtaining-data-and-evaluating-the-factualityprompt-benchmark)

## I. Quick Start with Interactive Mode

You can follow the steps below to quickly get up and running with Multi-expert Prompting.

1. **Clone and Download the Repository**

   ```bash
   git clone https://github.com/yourusername/Multi-expert-Prompting.git
   cd Multi-expert-Prompting
   ```

2. **Create and Activate a New Virtual Environment**

   ```bash
   conda create -n mep python=3.11
   conda activate mep
   ```

3. **Install Dependencies**

   In the top-level directory, run:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up OpenAI API Key (if using OpenAI models)**

   To run [OpenAI models](https://platform.openai.com/docs/models), you need to export your API key:

   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

5. **Run the Interactive Script**

   Use the following command:

    ```bash
    python src/interactive.py --model=[model] --num_experts=[number-of-experts] --temperature=[temperaure] [--verbose]
    ```

    Currently, we support the following open-source ([Mistral](https://huggingface.co/mistralai), [Meta-llama](https://huggingface.co/meta-llama)) and proprietary models ([OpenAI models](https://platform.openai.com/docs/models)):
      - --model: `gpt-4o`, `chatgpt-4o-latest`, `gpt-4o-2024-08-06`, `gpt-3.5-turbo`, `mistralai/Mistral-7B-Instruct-v0.2`, `meta-llama/Llama-3.1-8B-Instruct`.
      - --num_experts: any number. It is recommended to be less than 10 to avoid context window size exceedings. 
      - --temperature: often between 0 and 1.

    Example with `gpt-3.5-turbo` with 3 experts and temperature equal 0:

    ```
    python src/interactive.py --model="gpt-3.5-turbo" --num_experts=3 --temperature=0 --verbose
    ```

## II. Benchmark Experiment and Evaluation Scripts

### Running the TruthfulQA Benchmark

To evaluate truthfulness using the TruthfulQA benchmark, follow these steps:

#### Command

1. **Generate results using Multi-Expert Prompting**:
    ```bash
    python evaluation/benchmark/truthfulqa.py \
      --model [model] \
      --api_token [your-api-token] \
      --num_experts 3 \
      --verbose
    ```

2. **Evaluate results using the GPT-based judge**:
    ```bash
    python evaluation/metrics/truthfulqa_compute.py \
      --input_file [file obtained from above step] \
      --output_file [path to save evaluated results] \
      --judge_model [fine-tuned-model-name] \
      --api_token [your-api-token] \
      --run_judge
    ```

### Running the FactualityPrompt Benchmark

You need to obtain data from [FactualityPrompt evaluation guide](https://github.com/nayeon7lee/FactualityPrompt) and put into respective file.
To evaluate factual correctness using the FactualityPrompt benchmark:

#### Command

1. **Generate results using Multi-Expert Prompting**:
    ```bash
    python evaluation/benchmark/factualityprompt.py \
      --model [model] \
      --num_experts 3 \
      --prompt_type factual_250 \
      --api_token [your-api-key] \
      --verbose
    ```
**Note:** Replace `factual_250` with `nonfactual_250` for processing the non-factual subset in FactualityPrompt.


2. **Evaluate results**:  
   Refer to the official [FactualityPrompt evaluation guide](https://github.com/nayeon7lee/FactualityPrompt) for detailed instructions. Additionally, see the [Supplementary: Obtaining Data and Evaluating the FactualityPrompt Benchmark](#supplementary-obtaining-data-and-evaluating-the-factualityprompt-benchmark) section in this repository for a step-by-step walkthrough tailored to this project.

### Running the BOLD (Toxicity) Benchmark

#### Command

1. **Generate results using Multi-Expert Prompting**:
    ```bash
    python evaluation/benchmark/bold.py \
      --model [model] \
      --num_experts 3 \
      --api_token [your-api-key] \
      --verbose
    ```

2. **Compute toxicity scores**:
    ```bash
    python evaluation/metrics/toxicity_compute.py \
      --input_file [file obtained from above step] \
      --output_file [path to save computed toxicity scores]
    ```

### Running the HONEST Benchmark

To evaluate fairness using the HONEST benchmark:

#### Command

1. **Generate results using Multi-Expert Prompting**:
    ```bash
    python evaluation/benchmark/honest.py \
      --model [model] \
      --num_experts 3 \
      --api_token [your-api-key] \
      --verbose
    ```

2. **Compute HONEST scores**:
    ```bash
    python evaluation/metrics/HONEST_compute.py \
      --input_file [file obtained from above step] \
      --output_file [path to save computed HONEST scores]
    ```

---

## III. Main Results

The table below summarizes the performance of Multi-expert Prompting compared to several strong baselines. The details of our outputs are shared in the folder: `./evaluation/results`.

| **Mistral-7B-Inst. v0.2** | TruthfulQA ↑ | FactualityPrompt ↓ | BOLD ↓  | HONEST ↓ |
|---------------------------|--------------|--------------------|---------|----------|
| Zero-shot                  | 76.00        | 8.98/16.07         | **0.000**   | 0.012/0.009 |
| Zero-shot-CoT              | 78.70        | 9.28/14.87         | **0.000**   | 0.014/0.013 |
| Self-refine                | 81.88        | 10.36/14.95        | **0.000**   | 0.007/0.008 |
| Universal Self-consistency | 81.64        | 9.98/15.21         | **0.000**    | 0.007/0.008 |
| Multi-agent Debate         | 80.78        | 17.57/18.27        | **0.000**    | 0.004/0.007 |
| ExpertPrompting            | 80.34        | 11.43/15.32        | **0.000**   | 0.005/0.005 |
| **Multi-expert Prompting** | **87.15**    | **8.16/14.70**     | **0.000**   | **0.003/0.005** |

| **ChatGPT**                | TruthfulQA ↑ | FactualityPrompt ↓ | BOLD ↓  | HONEST ↓ |
|---------------------------|--------------|--------------------|---------|----------|
| Zero-shot                  | 68.05        | 6.99/12.90         | 0.163   | 0.038/0.023 |
| Zero-shot-CoT              | 70.38        | 6.93/13.75         | 0.163   | 0.006/0.005 |
| Self-refine                | 75.89        | 7.11/13.96         | 0.064   | 0.006/0.007 |
| Universal Self-consistency | 77.11        | 5.51/9.71          | **0.000**   | 0.010/0.008 |
| Multi-agent Debate         | 64.87        | 5.64/13.06         | **0.000**   | 0.005/0.004 |
| ExpertPrompting            | 80.66        | 5.64/15.66         | 0.129   | 0.004/0.004 |
| **Multi-expert Prompting** | **89.35**    | **4.54/9.45**      | **0.000**   | **0.004/0.003** |

**Key**: ↑ indicates higher is better; ↓ indicates lower is better.

## IV. Issues
Please report any software “bug”, or other problems with the models through one of the following means:

- GitHub [Issue Tracker](https://github.com/yourusername/Multi-expert-Prompting/issues).
- Email: [Do Xuan Long](mailto:xuanlong.do@u.nus.edu).

---

## V. Citation and Acknowledgements

If you find this repository helpful in your research, we appreciate your ⭐ and the paper citation:

```
@misc{long2024multiexpertpromptingimprovesreliability,
      title={Multi-expert Prompting Improves Reliability, Safety, and Usefulness of Large Language Models}, 
      author={Do Xuan Long and Duong Ngoc Yen and Anh Tuan Luu and Kenji Kawaguchi and Min-Yen Kan and Nancy F. Chen},
      year={2024},
      eprint={2411.00492},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.00492}, 
}
```

We would like to acknowledge the [Huggingface evaluate](https://github.com/huggingface/evaluate/tree/main) and [Huggingface transformers](https://github.com/huggingface/transformers).


## Supplementary: Fine-tuning the Judge Model for the TruthfulQA Benchmark

This guide provides step-by-step instructions on how to fine-tune a GPT-based judge model using the TruthfulQA dataset. After fine-tuning and obtaining the judge model, you will be able to evaluate the truthfulness of answers generated by language models. Instructions on how to run the TruthfulQA benchmark using the provided codebase are included in the [Running the TruthfulQA Benchmark](#running-the-truthfulqa-benchmark) section.

### Prerequisites

- Python 3.7 or higher
- An OpenAI API key with access to fine-tuning capabilities
- Necessary Python packages:
  - `openai`
  - `pandas`
  - `datasets`
  - `tqdm`
- Git (for cloning repositories)

### Downloading the Data

1. **Clone the TruthfulQA Repository:**

   ```bash
   git clone https://github.com/sylinrl/TruthfulQA.git
   ```

2. **Navigate to the Data Directory:**

   ```bash
   cd TruthfulQA/data
   ```

3. **Locate the Fine-tuning Data:**

   The fine-tuning data is provided in `finetune_truth.jsonl`. This file contains labeled examples for fine-tuning the judge model.

   Alternatively, you can download the file directly:

   ```bash
   wget https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/data/finetune_truth.jsonl
   ```

### Fine-tuning the Judge Model

We will fine-tune a GPT-based model (e.g., `gpt-3.5-turbo`) using the OpenAI API to create a judge model that can evaluate the truthfulness of answers.

#### 1. Prepare the Dataset

The fine-tuning dataset `finetune_truth.jsonl` is in JSON Lines format, where each line is a JSON object with the following structure:

```json
{
  "prompt": "Q: <question>\nA: <answer>\nTrue:",
  "completion": "<yes or no>"
}
```

Example:

```json
{
  "prompt": "Q: What is the capital of France?\nA: Paris.\nTrue:",
  "completion": "yes"
}
```

Ensure that the dataset is properly formatted and stored in a file accessible for fine-tuning.

#### 2. Fine-tune the Model with OpenAI API

Please refer to the official OpenAI fine-tuning guide for detailed instructions on how to fine-tune a model: [OpenAI Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)

**Note:** Fine-tuning capabilities are subject to OpenAI's policies and may require access approval. Be sure to comply with OpenAI's policies and monitor your usage in the OpenAI dashboard.

After fine-tuning is complete, you will receive a fine-tuned model name (e.g., `ft:gpt-3.5-turbo:your-org:2023-11-26-15-30-00`). Use this model as the `--judge_model` when running the TruthfulQA benchmark as described in the [Running the TruthfulQA Benchmark](#running-the-truthfulqa-benchmark) section.

### References

- **TruthfulQA Repository:** [https://github.com/sylinrl/TruthfulQA](https://github.com/sylinrl/TruthfulQA)
- **OpenAI Fine-tuning Guide:** [https://platform.openai.com/docs/guides/fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)
- **OpenAI API Reference:** [https://platform.openai.com/docs/api-reference/introduction](https://platform.openai.com/docs/api-reference/introduction)


## Supplementary: Obtaining Data and Evaluating the FactualityPrompt Benchmark

### Step 1: Clone the FactualityPrompt Repository

First, clone the official FactualityPrompt repository to access the dataset and evaluation scripts:

```bash
git clone https://github.com/nayeon7lee/FactualityPrompt.git
```

### Step 2: Obtain the Dataset

The FactualityPrompt dataset includes the following JSONL files:

- `fever_factual.jsonl`
- `fever_nonfactual.jsonl`

These datasets are located in the `prompts` directory of the cloned repository.

Navigate to the `prompts` directory:

```bash
cd FactualityPrompt/prompts
```

### Step 3: Copy the Dataset Files to Your Project

Copy the JSONL files to your project's data directory, such as `evaluation/data`:

```bash
mkdir -p /path/to/your/project/evaluation/data
cp fever_factual.jsonl /path/to/your/project/evaluation/data/
cp fever_nonfactual.jsonl /path/to/your/project/evaluation/data/
```

Replace `/path/to/your/project/` with the actual path to your project's root directory.

### Evaluating the Results

After generating responses using your model, you can evaluate the results using the FactualityPrompt evaluation scripts provided in the repository you cloned earlier.

#### Step 1: Install Required Dependencies

Navigate to the root directory of the cloned FactualityPrompt repository:

```bash
cd /path/to/FactualityPrompt
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

#### Step 2: Download and Prepare the Wikipedia Dump

The evaluation script requires access to a processed Wikipedia dump. Download the `kilt_knowledgesource.json` file from the KILT repository:

```bash
mkdir data
cd data
wget http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json
```

Create the database file (`kilt_db.db`) from the Wikipedia dump:

```bash
cd ..
PYTHONPATH=fever_athene python3 fever_athene/scripts/build_db_kilt.py data/kilt_knowledgesource.json data/kilt_db.db
```

#### Step 3: Configure `src/const.py`

Before running the evaluation scripts, configure the `const.py` file located in the `src` directory:

```bash
nano src/const.py
```

Update the paths in `const.py` to match your environment:

- `DB_PATH`: Set this to the path of the `kilt_db.db` file you just created (e.g., `data/kilt_db.db`).
- `DATA_PATH`: Ensure it points to the directory containing your data (e.g., `data/`).

Save and exit the editor.

#### Step 4: Run the Evaluation Scripts

##### Factuality Metrics (Hallucinated Named Entity Error, Entailment Ratio)

Run the evaluation script to compute the factuality metrics:

```bash
for PROMPT_TYPE in factual nonfactual
do
    GEN_TO_EVALUATE_NAME=${PROMPT_TYPE}-CUSTOM-GEN-NAME.jsonl
    PYTHONPATH=. python src/evaluate_v3_final.py --prompt_type ${PROMPT_TYPE} --gen_path ${GEN_TO_EVALUATE_NAME}
done
```

- Replace `CUSTOM-GEN-NAME` with the actual name of your generated file (without the prompt type prefix).
- This script will process both factual and non-factual prompts.
- The evaluation results will be saved in a file named `${GEN_TO_EVALUATE_NAME}_results.jsonl`.

**Example:**

If your generated file is named `factual-FactualityPrompt_MEP_3experts_mistral.jsonl`, run:

```bash
PROMPT_TYPE=factual
GEN_TO_EVALUATE_NAME=factual-FactualityPrompt_MEP_3experts_mistral.jsonl
PYTHONPATH=. python src/evaluate_v3_final.py --prompt_type $PROMPT_TYPE --gen_path $GEN_TO_EVALUATE_NAME
```

### References

- **FactualityPrompt Repository**: [https://github.com/nayeon7lee/FactualityPrompt](https://github.com/nayeon7lee/FactualityPrompt)
- **FEVER Dataset**: [https://fever.ai/](https://fever.ai/)
- **KILT Knowledge Source**: [https://github.com/facebookresearch/KILT](https://github.com/facebookresearch/KILT)
- **UKPLab FEVER Pipeline**: [https://github.com/UKPLab/fever-2018-team-athene](https://github.com/UKPLab/fever-2018-team-athene)