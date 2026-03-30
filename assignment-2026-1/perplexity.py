import argparse
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_log_probabilities(logits_row):
    """
    Υπολογίζει τις λογαριθμικές πιθανοφάνειες (log probabilities)
    για μια γραμμή logits, εφαρμόζοντας το trick για αποφυγή overflow.
    """
    row = logits_row.tolist()
    max_val = max(row)
    shifted = [x - max_val for x in row]

    # log(sum(exp(shifted)))
    log_sum_exp = math.log(sum(math.exp(x) for x in shifted))

    # log_probs[i] = shifted[i] - log_sum_exp
    return [x - log_sum_exp for x in shifted]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--begin-context-tokens", type=int, default=512)

    args = parser.parse_args()

    # Load model + tokenizer
    model_name = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, tie_word_embeddings=False
    )
    model.eval()

    bos_token = tokenizer.bos_token_id

    # Read input file
    with open(args.input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Tokenize
    tokens = tokenizer(text).input_ids

    # Sliding window parameters
    n_ctx = args.n_ctx
    stride = args.stride
    begin_ctx = args.begin_context_tokens

    total_neg_log_likelihood = 0.0
    total_predicted_tokens = 0

    # Process windows
    start = 0
    first_window = True

    while start < len(tokens):
        if first_window:
            # First window: only begin-context-tokens used as context
            window_tokens = [bos_token] + tokens[:begin_ctx]
            predict_start = 1  # predict from token 1 onward
            predict_end = len(window_tokens) - 1
            first_window = False
            start = begin_ctx
        else:
            end = min(start + n_ctx, len(tokens))
            window_tokens = [bos_token] + tokens[start:end]

            # Predict only the last "stride" tokens
            predict_start = len(window_tokens) - stride
            if predict_start < 1:
                predict_start = 1
            predict_end = len(window_tokens) - 1

            start += stride

        # Convert to tensor
        window_tensor = torch.tensor([window_tokens])

        with torch.no_grad():
            logits = model(window_tensor).logits  # shape: 1 x N x V

        # Compute log probabilities for each predicted token
        for i in range(predict_start, predict_end + 1):
            logits_row = logits[0, i - 1]  # prediction for token at position i
            log_probs = compute_log_probabilities(logits_row)

            true_token = window_tokens[i]
            log_prob = log_probs[true_token]

            total_neg_log_likelihood += -log_prob
            total_predicted_tokens += 1

    # Mean negative log likelihood
    mean_nll = total_neg_log_likelihood / total_predicted_tokens

    # Perplexity = exp(mean_nll)
    perplexity = math.exp(mean_nll)

    # Write output
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(f"{perplexity}\n")


if __name__ == "__main__":
    main()
