import torch
from transformers import GPT2TokenizerFast
from utils.setup_env import setup_environment
from utils.run_experiments import run_experiments
from models.gpt2_models import load_standard_gpt2, load_flash_attention_gpt2, load_flash_attention_with_alibi_gpt2
from dataset_loader.load_data import load_cnn_dataset, preprocess_function, get_input_summary
from train.fine_tuning import fine_tune_model


def main():
    setup_environment()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = 512
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset, train_dataset, val_dataset = load_cnn_dataset(1000, 200)

    # Apply preprocessing to the subsets
    tokenized_train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer=tokenizer, seq_len=512),
        batched=True
    )
    tokenized_val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer=tokenizer, seq_len=512),
        batched=True
    )

    # Input texts and reference summaries from validation dataset
    input_texts, reference_summaries = get_input_summary(dataset)

    # Load models
    model_standard = load_standard_gpt2(device)
    model_flash = load_flash_attention_gpt2(device)
    model_flash_with_alibi = load_flash_attention_with_alibi_gpt2(device)

    # Run the experiments
    run_experiments(model_standard, model_flash, model_flash_with_alibi, tokenizer, input_texts, reference_summaries, seq_len, device, 'benchmark_results')

    # Fine-tune Flash attention model with ALiBi, since GPT-2's pre-trained weights are not adjusted to ALiBi scores
    model_flash_with_alibi = fine_tune_model(model_flash_with_alibi, tokenizer, tokenized_train_dataset, tokenized_val_dataset)

    # Re-run the experiments
    run_experiments(model_standard, model_flash, model_flash_with_alibi, tokenizer, input_texts, reference_summaries, seq_len, device, 'benchmark_results_finetuned')


if __name__ == "__main__":
    main()
