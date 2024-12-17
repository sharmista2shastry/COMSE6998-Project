from .benchmarking import benchmark_model
from .evaluation import compute_metrics
from .generation import generate_text
from .tabulate_results import tabulate_results
from .wandb_logging import log_metrics_wandb


def run_experiments(model_standard, model_flash, model_flash_with_alibi, tokenizer, input_texts, reference_summaries, seq_len, device, exp_name):
    # Benchmark models
    standard_time, standard_mem = benchmark_model(model_standard, tokenizer, input_texts, device, seq_len)
    flash_time, flash_mem = benchmark_model(model_flash, tokenizer, input_texts)
    flash_alibi_time, flash_alibi_mem = benchmark_model(model_flash_with_alibi, tokenizer, input_texts)

    # Generate Texts
    standard_outputs = generate_text(model_standard, tokenizer, input_texts)
    flash_outputs = generate_text(model_flash, tokenizer, input_texts)
    flash_alibi_outputs = generate_text(model_flash_with_alibi, tokenizer, input_texts)

    # Compute Metrics
    standard_metrics = compute_metrics(standard_outputs, reference_summaries)
    flash_metrics = compute_metrics(flash_outputs, reference_summaries)
    flash_alibi_metrics = compute_metrics(flash_alibi_outputs, reference_summaries)

    time_data = (standard_time, flash_time, flash_alibi_time)
    mem_data = (standard_mem, flash_mem, flash_alibi_mem)
    metrics_data = (standard_metrics, flash_metrics, flash_alibi_metrics)

    # Print results in a tabulated manner
    tabulate_results(time_data, mem_data, metrics_data)

    # Log experiments in wandb
    log_metrics_wandb(time_data, mem_data, metrics_data, 'intro2llm', exp_name)