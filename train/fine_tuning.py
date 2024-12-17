from transformers import Trainer, TrainingArguments


def fine_tune_model(
    model,
    tokenizer,
    tokenized_train_dataset,
    tokenized_val_dataset
):
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./flash_alibi_summarization",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-5,
        warmup_steps=500,
        weight_decay=0.01,
        save_steps=1000,
        evaluation_strategy="steps",
        eval_steps=1000,
        logging_dir="./logs",
        logging_steps=500,
        save_total_limit=2,
        gradient_accumulation_steps=4,
        fp16=True,  # Enable mixed precision for faster training if supported
    )

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
    )

    # Start training
    trainer.train()

    return model
