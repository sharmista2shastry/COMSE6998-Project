# Enhancing FlashAttention with ALiBi in Transformer Models

`Team: Sharmista Shastry (ss6950), Rishabh Srivastava (rs4489)`

This repository demonstrates the integration of ALiBi (Attention with Linear Biases) scores in FlashAttention to enhance performance. We used GPT-2 models to test this integration, comparing the resulting inference efficiency and text generation quality with standard attention and FlashAttention variants. The main goal was to explore how ALiBi can improve the performance of FlashAttention in language modeling tasks, particularly text summarization.

## Requirements

- Python
- PyTorch
- Hugging Face Transformers
- FlashAttention
- Datasets
- Wandb

First, install the flash attention (`flash_attn`) module by running the following commands:

```bash
git clone https://github.com/HazyResearch/flash-attention.git
cd flash-attention
git checkout v2.7.2.post1
git submodule update --init --recursive
pip install . --no-build-isolation
```

Install the necessary packages via pip:

```bash
pip install torch transformers datasets wandb nltk rouge-score tabulate
```

Since Flash Attention needs Ampere GPUs and above, so considerable GPU compute resources (A100 and above) are required.

## Dataset

We use the **CNN/DailyMail** dataset (version 3.0.0) for text summarization tasks. The dataset is split into training and validation subsets.

## Models

The model is configured with three variants:
1. Standard Attention in GPT-2
2. Flash Attention in GPT-2
3. Flash Attention with ALiBi in GPT-2

We also fine-tune the last GPT-2 model that has Flash Attention with ALiBi scores, since GPT-2's pre-trained weights are not adjusted to attention with ALiBi scores. 

## Running the Experiments
1. You can run the `main.py` script by this command:
```bash
python main.py
```
2. The code has also been presented in the Python Notebook `FlashAttn_with_ALiBi.ipynb` for an easier comprehension of the code flow.

## Results

The benchmarking results are displayed in two tables: **Efficiency Comparison** and **Quality Comparison**.

The efficiency comparison includes the inference time (in milliseconds) and memory usage (in MB) for each model.

<figure>
    <figcaption><b>Efficiency Metrics for Different Experiments</b></figcaption>
    <img src="assets/tab1.png"
         alt="Efficiency Metrics">
</figure>

<figure>
    <figcaption><b>Efficiency Metrics: Scatter Plot</b></figcaption>
    <img src="assets/graph.png"
         alt="Efficiency Metrics Scatter Plot">
</figure>

The quality comparison includes ROUGE-1, ROUGE-2, ROUGE-L, and BLEU scores for each model.

<figure>
    <figcaption><b>Quality Comparison: ROUGE-1</b></figcaption>
    <img src="assets/ROUGE1.png"
         alt="Quality Comparison ROUGE1">
</figure>

<figure>
    <figcaption><b>Quality Comparison: ROUGE-2</b></figcaption>
    <img src="assets/ROUGE2.png"
         alt="Quality Comparison ROUGE2">
</figure>

<figure>
    <figcaption><b>Quality Comparison: ROUGE-L</b></figcaption>
    <img src="assets/ROUGEL.png"
         alt="Quality Comparison ROUGEL">
</figure>

<figure>
    <figcaption><b>Quality Comparison: BLEU</b></figcaption>
    <img src="assets/BLEU.png"
         alt="Quality Comparison BLEU">
</figure>

You can find the **Wandb report** with performance benchamarking [here](https://wandb.ai/hpmlcolumbia/intro2llm/reports/Performance-and-Quality-Benchmarking-of-Standard-GPT-2-Flash-Attention-GPT-2-and-Flash-Attention-ALiBi-GPT-2--VmlldzoxMDYxMDkyMA?accessToken=a69c5flkf4r4r2bwepgfxfraj06efcyjs2z3actqn4uoj4hd0ye6my8lkltifc2r).



