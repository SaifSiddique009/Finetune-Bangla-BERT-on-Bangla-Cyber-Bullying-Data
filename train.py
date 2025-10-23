"""
Training module with enhanced metrics, MLflow tracking, and improved logging
Includes both weighted and macro metrics, training set evaluation, and optimized MLflow structure
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
import mlflow

import data
from model import TransformerMultiLabelClassifier
from utils import get_model_metrics, print_fold_summary, print_experiment_summary


def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive metrics for multi-label classification.
    Includes both weighted and macro averages for better insight into model performance.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities (will be thresholded at 0.5)
        
    Returns:
        dict: Dictionary containing all metrics
    """
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    return {
        'accuracy': accuracy_score(y_true, y_pred_binary),
        
        # Weighted metrics - gives more importance to frequent classes
        'precision_weighted': precision_score(y_true, y_pred_binary, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred_binary, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred_binary, average='weighted', zero_division=0),
        
        # Macro metrics - treats all classes equally (better for imbalance insight)
        'precision_macro': precision_score(y_true, y_pred_binary, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred_binary, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred_binary, average='macro', zero_division=0),
    }


def train_epoch(model, dataloader, optimizer, scheduler, device, class_weights=None, max_norm=1.0):
    """
    Train the model for one epoch and calculate training metrics.
    
    Args:
        model: The transformer model
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to run training on
        class_weights: Optional class weights for imbalanced data
        
    Returns:
        dict: Training metrics including loss and performance metrics
    """
    model.train()
    total_loss = 0
    all_train_predictions = []
    all_train_labels = []
    
    # Setup loss function with class weights if provided
    if class_weights is not None:
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
    else:
        loss_fct = nn.BCEWithLogitsLoss()
    
    for batch in tqdm(dataloader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass without computing loss in model (to use custom loss with weights)
        outputs = model(input_ids, attention_mask=attention_mask, labels=None)
        
        # Calculate loss with optional class weights
        loss = loss_fct(outputs['logits'], labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Accumulate predictions for metrics calculation (detached from computation graph)
        with torch.no_grad():
            predictions = torch.sigmoid(outputs['logits'])
            all_train_predictions.extend(predictions.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
    
    # Calculate training metrics
    avg_loss = total_loss / len(dataloader)
    train_metrics = calculate_metrics(np.array(all_train_labels), np.array(all_train_predictions))
    train_metrics['loss'] = avg_loss
    
    return train_metrics


def evaluate_model(model, dataloader, device, class_weights=None):
    """
    Evaluate the model on validation data.
    
    Args:
        model: The transformer model
        dataloader: Validation data loader
        device: Device to run evaluation on
        class_weights: Optional class weights for loss calculation
        
    Returns:
        dict: Validation metrics including loss and performance metrics
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    # Setup loss function
    if class_weights is not None:
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
    else:
        loss_fct = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=None)
            loss = loss_fct(outputs['logits'], labels)
            
            total_loss += loss.item()
            
            predictions = torch.sigmoid(outputs['logits'])
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_predictions))
    metrics['loss'] = avg_loss
    
    return metrics


def print_epoch_metrics(epoch, num_epochs, fold, num_folds, train_metrics, val_metrics, best_f1, best_epoch):
    """
    Print comprehensive epoch metrics in a formatted way.
    
    Args:
        epoch: Current epoch (0-indexed)
        num_epochs: Total number of epochs
        fold: Current fold (0-indexed)
        num_folds: Total number of folds
        train_metrics: Training metrics dictionary
        val_metrics: Validation metrics dictionary
        best_f1: Best validation F1 score so far
        best_epoch: Epoch with best F1 score (0-indexed)
    """
    print("\n" + "="*60)
    print(f"Epoch {epoch+1}/{num_epochs} | Fold {fold+1}/{num_folds}")
    print("="*60)
    
    print("TRAINING:")
    print(f"  Loss: {train_metrics['loss']:.4f}")
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  Precision (weighted): {train_metrics['precision_weighted']:.4f} | (macro): {train_metrics['precision_macro']:.4f}")
    print(f"  Recall (weighted): {train_metrics['recall_weighted']:.4f} | (macro): {train_metrics['recall_macro']:.4f}")
    print(f"  F1 (weighted): {train_metrics['f1_weighted']:.4f} | (macro): {train_metrics['f1_macro']:.4f}")
    
    print("\nVALIDATION:")
    print(f"  Loss: {val_metrics['loss']:.4f}")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Precision (weighted): {val_metrics['precision_weighted']:.4f} | (macro): {val_metrics['precision_macro']:.4f}")
    print(f"  Recall (weighted): {val_metrics['recall_weighted']:.4f} | (macro): {val_metrics['recall_macro']:.4f}")
    print(f"  F1 (weighted): {val_metrics['f1_weighted']:.4f} | (macro): {val_metrics['f1_macro']:.4f}")
    
    print(f"\n⭐ Best F1 so far: {best_f1:.4f} (Epoch {best_epoch})")
    print("="*60)


def run_kfold_training(config, comments, labels, tokenizer, device):
    """
    Run K-fold cross-validation training with enhanced metrics and logging.
    
    Args:
        config: Configuration object with all hyperparameters
        comments: Array of text comments
        labels: Array of multi-label targets
        tokenizer: Tokenizer for text encoding
        device: Device to run training on
    """
    # Set up MLflow experiment
    mlflow.set_experiment(config.mlflow_experiment_name)
    
    with mlflow.start_run(run_name=f"{config.author_name}_batch{config.batch}_lr{config.lr}_epochs{config.epochs}"):
        
        # Log all configuration parameters
        mlflow.log_params({
            'batch_size': config.batch,
            'learning_rate': config.lr,
            'num_epochs': config.epochs,
            'num_folds': config.num_folds,
            'max_length': config.max_length,
            'freeze_base': config.freeze_base,
            'dropout': config.dropout,
            'weight_decay': config.weight_decay,
            'warmup_ratio': config.warmup_ratio,
            'gradient_clip_norm': config.gradient_clip_norm,
            'early_stopping_patience': config.early_stopping_patience,
            'author_name': config.author_name,
            'model_path': config.model_path,
            'seed': config.seed,
            'stratification_type': config.stratification_type
        })
        
        # Prepare K-fold splits with stratification
        kfold_splits = data.prepare_kfold_splits(
            comments, labels, 
            num_folds=config.num_folds,
            stratification_type=config.stratification_type,
            seed=config.seed
        )
        
        # Store results for each fold
        fold_results = []
        best_fold_model = None
        best_fold_idx = -1
        best_overall_f1 = 0
        
        for fold, (train_idx, val_idx) in enumerate(kfold_splits):
            print(f"\n{'='*60}")
            print(f"FOLD {fold + 1}/{config.num_folds}")
            print('='*60)
            
            # Split data for current fold
            train_comments, val_comments = comments[train_idx], comments[val_idx]
            train_labels, val_labels = labels[train_idx], labels[val_idx]
            
            # Calculate class weights for imbalanced data
            class_weights = data.calculate_class_weights(train_labels)
            
            # Create datasets and dataloaders
            train_dataset = data.CyberbullyingDataset(train_comments, train_labels, tokenizer, config.max_length)
            val_dataset = data.CyberbullyingDataset(val_comments, val_labels, tokenizer, config.max_length)
            
            train_loader = DataLoader(train_dataset, batch_size=config.batch, shuffle=True, num_workers=2, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=config.batch, shuffle=False, num_workers=2, pin_memory=True)
            
            # Initialize model
            model = TransformerMultiLabelClassifier(config.model_path, len(data.LABEL_COLUMNS), dropout=config.dropout)
            
            # Freeze base layers if specified
            if config.freeze_base:
                model.freeze_base_layers()
            
            model.to(device)
            
            # Get model metrics (only once per experiment, not per fold)
            if fold == 0:
                model_metrics = get_model_metrics(model)
                mlflow.log_metrics({
                    'total_parameters': model_metrics['total_parameters'],
                    'trainable_parameters': model_metrics['trainable_parameters'],
                    'model_size_mb': model_metrics['model_size_mb']
                })
            
            # Setup optimizer and scheduler
            optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, eps=1e-8)
            total_steps = len(train_loader) * config.epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=int(config.warmup_ratio * total_steps), 
                num_training_steps=total_steps
            )
            
            # Training loop variables
            best_f1 = 0
            best_metrics = {}
            best_epoch = 0
            patience = config.early_stopping_patience
            patience_counter = 0
            
            for epoch in range(config.epochs):
                # Train for one epoch
                train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, class_weights, max_norm=config.gradient_clip_norm)
                
                # Evaluate on validation set
                val_metrics = evaluate_model(model, val_loader, device, class_weights)
                               
                # Check if this is the best epoch for this fold
                if val_metrics['f1_weighted'] > best_f1:
                    best_f1 = val_metrics['f1_weighted']
                    best_metrics = val_metrics.copy()
                    best_metrics.update({f'train_{k}': v for k, v in train_metrics.items()})
                    best_epoch = epoch+1
                    patience_counter = 0
                    
                    # Save model if this fold is the best overall
                    if best_f1 > best_overall_f1:
                        best_overall_f1 = best_f1
                        best_fold_idx = fold
                        best_fold_model = model.state_dict()
                else:
                    patience_counter += 1
                
                # Print comprehensive metrics
                print_epoch_metrics(epoch, config.epochs, fold, config.num_folds, 
                                  train_metrics, val_metrics, best_f1, best_epoch)

                # Early stopping
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
            
            # Store best metrics for this fold
            best_metrics['best_epoch'] = best_epoch
            fold_results.append(best_metrics)
            
            # Log best metrics for this fold to MLflow (not epoch-level metrics)
            for metric_name, metric_value in best_metrics.items():
                if not metric_name.startswith('train_'):
                    mlflow.log_metric(f"fold_{fold+1}_best_{metric_name}", metric_value)
            
            # Print fold summary
            print_fold_summary(fold, best_metrics, best_epoch)
        
        # Find best fold based on weighted F1
        best_fold_metrics = fold_results[best_fold_idx]
        
        # Log best overall metrics (maximum across all folds)
        mlflow.log_metric('best_fold_index', best_fold_idx+1)
        
        # Log the best value for each metric across all folds
        for metric_name in ['accuracy', 'precision_weighted', 'precision_macro', 
                           'recall_weighted', 'recall_macro', 'f1_weighted', 'f1_macro']:
            best_value = max([fold_result[metric_name] for fold_result in fold_results])
            mlflow.log_metric(f'best_{metric_name}', best_value)
        
        # Log minimum loss
        best_loss = min([fold_result['loss'] for fold_result in fold_results])
        mlflow.log_metric('best_loss', best_loss)
        
        # Save the best model from the best fold
        # if best_fold_model is not None:
        #     # Create a new model instance and load the best weights
        #     final_model = TransformerMultiLabelClassifier(config.model_path, len(data.LABEL_COLUMNS), dropout=config.dropout)
        #     final_model.load_state_dict(best_fold_model)
            
        #     # Save model with MLflow
        #     mlflow.pytorch.log_model(
        #         final_model, 
        #         name="model",
        #         registered_model_name=f"bangla_cyberbully_model_fold{best_fold_idx+1}_f1_{best_overall_f1:.4f}"
        #     )
            
        #     # Also save locally
        #     model_filename = f"best_model_fold_{best_fold_idx+1}_f1_{best_overall_f1:.4f}.pt"
        #     torch.save(best_fold_model, model_filename)
        #     print(f"\nModel saved: {model_filename}")
        
        # Print final experiment summary
        print_experiment_summary(best_fold_idx, best_fold_metrics, model_metrics)