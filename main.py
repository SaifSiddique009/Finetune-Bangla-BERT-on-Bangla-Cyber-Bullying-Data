# main.py
import torch
from transformers import BertTokenizer
import data
import model
import train
from config import parse_arguments

def main():
    config = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained(config.model_path)

    comments, labels = data.load_and_preprocess_data(config.dataset_path)
    print(f"Loaded {len(comments)} samples with {len(data.LABEL_COLUMNS)} labels.")

    train.run_kfold_training(config, comments, labels, tokenizer, device)

if __name__ == "__main__":
    main()
