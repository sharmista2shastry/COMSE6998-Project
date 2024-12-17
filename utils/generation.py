import torch
from torch.amp import autocast


def generate_text(model, tokenizer, texts, seq_len=512, device='cpu', max_new_tokens=100, num_beams=5):
    outputs = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True, padding=True).to(device)
        attention_mask = inputs["attention_mask"]

        model.config.pad_token_id = tokenizer.eos_token_id

        with torch.no_grad(), autocast(device_type='cuda'):
            output_ids = model.generate(
                inputs["input_ids"],
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )
        outputs.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    return outputs
