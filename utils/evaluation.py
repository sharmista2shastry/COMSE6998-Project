from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def compute_metrics(predicted, references):
    """Compute ROUGE and BLEU metrics."""
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    bleu_score = 0

    for pred, ref in zip(predicted, references):
        rouge_res = rouge.score(pred, ref)
        rouge_scores["rouge1"] += rouge_res["rouge1"].fmeasure
        rouge_scores["rouge2"] += rouge_res["rouge2"].fmeasure
        rouge_scores["rougeL"] += rouge_res["rougeL"].fmeasure

        smoothing_fn = SmoothingFunction().method4
        bleu_score += sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothing_fn)

    for key in rouge_scores:
        rouge_scores[key] /= len(predicted)
    bleu_score /= len(predicted)
    return rouge_scores, bleu_score
