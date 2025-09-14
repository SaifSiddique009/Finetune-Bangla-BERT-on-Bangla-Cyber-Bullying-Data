import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune BanglaBERT for multi-label cyberbullying detection.")
    parser.add_argument('--batch', type=int, default=16, help='Batch size for training and evaluation.')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate for optimizer.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--author_name', type=str, required=True, help='Author name for MLflow run tagging.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the CSV dataset file.')
    parser.add_argument('--model_path', type=str, default='sagorsarker/bangla-bert-base', help='Pre-trained model path.')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for K-Fold cross-validation.')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length for tokenization.')
    parser.add_argument('--freeze_base', action='store_true', help='Freeze the base BERT layers during fine-tuning.')
    parser.add_argument('--mlflow_experiment_name', type=str, default='Bangla-BERT-Cyberbullying', help='MLflow experiment name.')
    return parser.parse_args()
