
[![arXiv](https://img.shields.io/badge/arXiv-2411.00492-b31b1b.svg)](https://arxiv.org/abs/2411.00492)


<div align="left">

<h1>[EMNLP 2024]  Multi-expert Prompting Improves Reliability, Safety, and Usefulness of Large Language Models</h1>
<!-- <div>
    <a href='https://dxlong2000.github.io/' target='_blank'>Do Xuan Long</a><sup>1,2</sup>&emsp;
    <a>Duong Ngoc Yen</a><sup>3</sup>&emsp;
    <a href='https://tuanluu.github.io/' target='_blank'>Luu Anh Tuan</a><sup>3</sup>&emsp;
    <a href='https://ml.comp.nus.edu.sg/#members' target='_blank'>Kenji Kawaguchi</a><sup>1</sup>&emsp;
    <a href='https://www.comp.nus.edu.sg/~kanmy/' target='_blank'>Min-Yen Kan</a><sup>1</sup>&emsp;
    <a href='https://sites.google.com/site/nancyfchen/home' target='_blank'>Nancy F. Chen</a><sup>2</sup>&emsp;
</div>
<div>
    <sup>1</sup>National University of Singapore,&emsp;<br>
    <sup>2</sup>Institute for Infocomm Research (I2R), A*STAR,&emsp;<br>
    <sup>3</sup>Nanyang Technological University&emsp;
</div>
</div> -->



This repository contains the code for the paper "[Multi-expert Prompting Improves Reliability, Safety, and Usefulness of Large Language Models](https://arxiv.org/abs/2411.00492)". Below is its workflow.

<img src="images/overview.png" width="95%"/>

<!-- , an innovative method to improve the generation quality of Large Language Models (LLMs) by simulating multiple expert perspectives, aggregating their responses, and selecting the most accurate and useful answers. Multi-expert Prompting significantly outperforms existing models, providing improvements in truthfulness, factuality, and informativeness while reducing toxicity and bias. -->
<!-- 
## Paper

- **Title**: Multi-expert Prompting Improves Reliability, Safety, and Usefulness of Large Language Models
- **Authors**: Do Xuan Long, Duong Ngoc Yen, Luu Anh Tuan, Kenji Kawaguchi, Min-Yen Kan, Nancy F. Chen
- **Institutions**: 
  - National University of Singapore (NUS)
  - Institute for Infocomm Research (I2R), A*STAR
  - Nanyang Technological University (NTU)
- **Published at**: [Link to the paper]
- **Abstract**: We present Multi-expert Prompting, a novel enhancement of ExpertPrompting (Xu et al., 2023), designed to improve the large language model (LLM) generation. Specifically, it guides an LLM to fulfill an input instruction by simulating multiple experts, aggregating their responses, and selecting the best among individual and aggregated responses. This process is performed in a single chain of thoughts through our seven carefully designed subtasks derived from the Nominal Group Technique (Ven and Delbecq, 1974), a well-established decision-making framework. Our evaluations demonstrate that Multi-expert Prompting significantly outperforms ExpertPrompting and comparable baselines in enhancing the truthfulness, factuality, informativeness, and usefulness of responses while reducing toxicity and hurtfulness. It further achieves state-of-the-art truthfulness by outperforming the best baseline by 8.69% with ChatGPT. Multi-expert Prompting is efficient, explainable, and highly adaptable to diverse scenarios, eliminating the need for manual prompt construction. -->

## I. Quick Start with Interactive Mode

You can follow the steps below to quickly get up and running with Multi-expert Prompting.

1. In a conda env with PyTorch / CUDA available clone and download this repository.

2. Create and activate a new virtual environment.

    ```bash
    conda create -n mep python=3.11
    conda activate mep
    ```

3. In the top-level directory run:
    ```bash
    pip install -r requirements.txt
    ```
4. To run [OpenAI models](https://platform.openai.com/docs/models), you need to export your API key:
    ```
    export OPENAI_API_KEY=your_api_key_here
    ```
4. Once you got everything installed correctly, use the following command:

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

**Benchmark experiments:** *Benchmarking data and scripts are coming soon! Alternatively, you can shortly customize `src/interactive.py` to run your own benchmark experiments.*

**Benchmark evaluations:** We share our outputs in the folder: `./evaluation/results`. To obtain the evaluation results, perform the following steps:

1.  Navigate to the directory `metrics`.

    ```
    cd Multi-expert-Prompting/evaluation/metrics
    ```
2. Run the scripts there to compute metrics:
    ```
    python BOLD_compute.py
    python TOXICITY_compute.py
    python HONEST_compute.py
    ```

    *Note: Evaluation instructions for TruthfulQA, FactualityPrompt and ExpertQA are coming soon!*

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

- This Github repo.
- Do Xuan Long via xuanlong.do@u.nus.edu.

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

