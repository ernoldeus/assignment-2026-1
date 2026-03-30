import argparse
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--begin-context-tokens", type=int, default=512)




if __name__ == "__main__":
    main()
