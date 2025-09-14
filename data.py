import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from transformers import BertTokenizer
from torch.utils.data import Dataset
import numpy as np
import torch

LABEL_COLUMNS = ['bully', 'sexual', 'religious', 'threat', 'spam']

class CyberbullyingDataset(Dataset):
    def __init__(self, comments, labels, tokenizer, max_length=128):
        self.comments = comments
        self.labels = labels.astype(np.float32) if isinstance(labels, np.ndarray) else labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = str(self.comments[idx])
        labels = self.labels[idx]

        encoding = self.tokenizer(
            comment,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

def load_and_preprocess_data(dataset_path):
    df = pd.read_csv(dataset_path)
    # Drop unnecessary columns if present
    columns_to_drop = [col for col in ['Gender', 'Profession'] if col in df.columns]
    df_clean = df.drop(columns_to_drop, axis=1) if columns_to_drop else df
    # Ensure label columns exist
    for col in LABEL_COLUMNS:
        if col not in df_clean.columns:
            raise ValueError(f"Missing label column: {col}")
    comments = df_clean['comment'].values
    labels = df_clean[LABEL_COLUMNS].values
    return comments, labels

def prepare_kfold_splits(comments, labels, num_folds=5):
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    return kfold.split(comments)
