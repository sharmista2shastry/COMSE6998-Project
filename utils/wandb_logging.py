import wandb


def log_metrics_wandb(time_data, mem_data, metrics_data, project_name='intro2llm', run_name='benchmark_results'):
    standard_time, flash_time, flash_alibi_time = time_data
    standard_mem, flash_mem, flash_alibi_mem = mem_data
    standard_metrics, flash_metrics, flash_alibi_metrics = metrics_data

    # Initialize wandb
    wandb.init(project=project_name, name=run_name)

    # Log efficiency results
    efficiency_table = wandb.Table(columns=["Model", "Inference Time (ms)", "Memory (MB)"])
    efficiency_table.add_data("Standard GPT-2", standard_time, standard_mem)
    efficiency_table.add_data("Flash GPT-2", flash_time, flash_mem)
    efficiency_table.add_data("Flash + ALiBi GPT-2", flash_alibi_time, flash_alibi_mem)
    wandb.log({"Efficiency Metrics": efficiency_table})

    # Log quality results
    quality_table = wandb.Table(columns=["Model", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"])
    quality_table.add_data("Standard GPT-2",
                           standard_metrics[0]['rouge1'],
                           standard_metrics[0]['rouge2'],
                           standard_metrics[0]['rougeL'],
                           standard_metrics[1])
    quality_table.add_data("Flash GPT-2",
                           flash_metrics[0]['rouge1'],
                           flash_metrics[0]['rouge2'],
                           flash_metrics[0]['rougeL'],
                           flash_metrics[1])
    quality_table.add_data("Flash + ALiBi GPT-2",
                           flash_alibi_metrics[0]['rouge1'],
                           flash_alibi_metrics[0]['rouge2'],
                           flash_alibi_metrics[0]['rougeL'],
                           flash_alibi_metrics[1])
    wandb.log({"Quality Metrics": quality_table})

    # Log as summary
    wandb.summary["Standard Inference Time (ms)"] = standard_time
    wandb.summary["Flash Inference Time (ms)"] = flash_time
    wandb.summary["Flash + ALiBi Inference Time (ms)"] = flash_alibi_time

    wandb.summary["Standard Memory (MB)"] = standard_mem
    wandb.summary["Flash Memory (MB)"] = flash_mem
    wandb.summary["Flash + ALiBi Memory (MB)"] = flash_alibi_mem

    wandb.summary["Standard ROUGE-1"] = standard_metrics[0]['rouge1']
    wandb.summary["Flash ROUGE-1"] = flash_metrics[0]['rouge1']
    wandb.summary["Flash + ALiBi ROUGE-1"] = flash_alibi_metrics[0]['rouge1']

    wandb.summary["Standard ROUGE-2"] = standard_metrics[0]['rouge2']
    wandb.summary["Flash ROUGE-2"] = flash_metrics[0]['rouge2']
    wandb.summary["Flash + ALiBi ROUGE-2"] = flash_alibi_metrics[0]['rouge2']

    wandb.summary["Standard ROUGE-L"] = standard_metrics[0]['rougeL']
    wandb.summary["Flash ROUGE-L"] = flash_metrics[0]['rougeL']
    wandb.summary["Flash + ALiBi ROUGE-L"] = flash_alibi_metrics[0]['rougeL']

    wandb.summary["Standard BLEU"] = standard_metrics[1]
    wandb.summary["Flash BLEU"] = flash_metrics[1]
    wandb.summary["Flash + ALiBi BLEU"] = flash_alibi_metrics[1]

    wandb.finish()