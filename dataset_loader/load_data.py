from datasets import load_dataset


# Preprocess dataset
def preprocess_function(example, tokenizer, seq_len=512):
    input_text = example["article"]
    target_text = example["highlights"]

    # Tokenize the input text with the updated sequence length
    model_inputs = tokenizer(
        input_text, max_length=seq_len, truncation=True, padding="max_length"
    )

    # Tokenize the target text
    labels = tokenizer(
        target_text, max_length=seq_len, truncation=True, padding="max_length"
    )

    # Mask padding tokens in labels to -100 so they are ignored by the loss function
    labels["input_ids"] = [
        (label if label != tokenizer.pad_token_id else -100) for label in labels["input_ids"]
    ]

    # Add labels to the model inputs
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def load_cnn_dataset(train_subset_size=1000, val_subset_size=200):
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    train_dataset = dataset["train"].shuffle(seed=42).select(range(train_subset_size))
    val_dataset = dataset["validation"].shuffle(seed=42).select(range(val_subset_size))
    return dataset, train_dataset, val_dataset


def get_input_summary(dataset):
    input_texts = [example["article"] for example in dataset["validation"].select(range(5))]
    reference_summaries = [example["highlights"] for example in dataset["validation"].select(range(5))]
    return input_texts, reference_summaries
