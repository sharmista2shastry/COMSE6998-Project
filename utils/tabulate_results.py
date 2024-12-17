from tabulate import tabulate


def tabulate_results(time_data, mem_data, metrics_data):
    standard_time, flash_time, flash_alibi_time = time_data
    standard_mem, flash_mem, flash_alibi_mem = mem_data
    standard_metrics, flash_metrics, flash_alibi_metrics = metrics_data

    results_efficiency = [
        ["Model", "Inference Time (ms)", "Memory (MB)"],
        ["Standard GPT-2", f"{standard_time:.2f}", f"{standard_mem:.2f}"],
        ["Flash GPT-2", f"{flash_time:.2f}", f"{flash_mem:.2f}"],
        ["Flash + ALiBi GPT-2", f"{flash_alibi_time:.2f}", f"{flash_alibi_mem:.2f}"]
    ]

    results_quality = [
        ["Model", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"],
        ["Standard GPT-2", f"{standard_metrics[0]['rouge1']:.4f}", f"{standard_metrics[0]['rouge2']:.4f}",
         f"{standard_metrics[0]['rougeL']:.4f}", f"{standard_metrics[1]:.4f}"],
        ["Flash GPT-2", f"{flash_metrics[0]['rouge1']:.4f}", f"{flash_metrics[0]['rouge2']:.4f}",
         f"{flash_metrics[0]['rougeL']:.4f}", f"{flash_metrics[1]:.4f}"],
        ["Flash + ALiBi GPT-2", f"{flash_alibi_metrics[0]['rouge1']:.4f}", f"{flash_alibi_metrics[0]['rouge2']:.4f}",
         f"{flash_alibi_metrics[0]['rougeL']:.4f}", f"{flash_alibi_metrics[1]:.4f}"]
    ]

    print("\n===== Efficiency Comparison =====\n")
    print(tabulate(results_efficiency, headers="firstrow", tablefmt="pretty"))

    print("\n===== Quality Comparison =====\n")
    print(tabulate(results_quality, headers="firstrow", tablefmt="pretty"))

    return