import argparse
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_log_probabilities(logits_row):
    row = logits_row.tolist()
    max_val = max(row)
    shifted = [x - max_val for x in row]
    log_sum_exp = math.log(sum(math.exp(x) for x in shifted))
    return [x - log_sum_exp for x in shifted]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--begin-context-tokens", type=int, default=512)

    args = parser.parse_args()
    model_name = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, tie_word_embeddings=False
    )
    model.eval()
    bos_token = tokenizer.bos_token_id
    with open(args.input_file, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenizer(text).input_ids
    n_ctx = args.n_ctx
    stride = args.stride
    begin_ctx = args.begin_context_tokens

    total_neg_log_likelihood = 0.0
    total_predicted_tokens = 0
    start = 0
    first_window = True

    while start < len(tokens):
        if first_window:
            window_tokens = [bos_token] + tokens[:begin_ctx]
            predict_start = 1 
            predict_end = len(window_tokens) - 1
            first_window = False
            start = begin_ctx
        else:
            end = min(start + n_ctx, len(tokens))
            window_tokens = [bos_token] + tokens[start:end]
            predict_start = len(window_tokens) - stride
            if predict_start < 1:
                predict_start = 1
            predict_end = len(window_tokens) - 1

            start += stride
        window_tensor = torch.tensor([window_tokens])

        with torch.no_grad():
            logits = model(window_tensor).logits  
        for i in range(predict_start, predict_end + 1):
            logits_row = logits[0, i - 1]  
            log_probs = compute_log_probabilities(logits_row)
            true_token = window_tokens[i]
            log_prob = log_probs[true_token]
            total_neg_log_likelihood += -log_prob
            total_predicted_tokens += 1

    mean_nll = total_neg_log_likelihood / total_predicted_tokens
    perplexity = math.exp(mean_nll)
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(f"{perplexity}\n")


if __name__ == "__main__":
    main()
