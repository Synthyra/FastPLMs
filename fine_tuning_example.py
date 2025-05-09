#! /usr/bin/env python3
"""
This script is a simple example of how to fine-tune a Synthyra FastPLM model for a protein sequence regression or classification task.
For regression we look at the binding affinity of two proteins (pkd)
For classification we look at the solubility of a protein (membrane bound or not)
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset
from typing import List, Tuple, Dict, Union, Any
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    EvalPrediction
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import spearmanr


# Shared arguments for the trainer
BASE_TRAINER_KWARGS = {
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "logging_steps": 100,
    "eval_strategy": "steps",
    "eval_steps": 500,
    "save_strategy": "steps",
    "save_steps": 500,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "report_to": "none",
    "label_names": ["labels"]
}


# Dataset classes
class PairDatasetHF(TorchDataset):
    """
    Dataset class for protein pair data (e.g., protein-protein interactions).
    
    Args:
        data: The dataset containing protein sequences and labels
        col_a: Column name for the first protein sequence
        col_b: Column name for the second protein sequence
        label_col: Column name for the labels
        max_length: Maximum sequence length to consider
    """
    def __init__(self, dataset: Any, col_a: str, col_b: str, label_col: str, max_length: int = 2048):
        self.seqs_a = dataset[col_a]
        self.seqs_b = dataset[col_b]
        self.labels = dataset[label_col]
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.seqs_a)

    def __getitem__(self, idx: int) -> Tuple[str, str, Union[float, int]]:
        seq_a = self.seqs_a[idx][:self.max_length]
        seq_b = self.seqs_b[idx][:self.max_length]
        label = self.labels[idx]
        return seq_a, seq_b, label


class SequenceDatasetHF(TorchDataset):
    """
    Dataset class for single protein sequence data.
    
    Args:
        dataset: The dataset containing protein sequences and labels
        col_name: Column name for the protein sequences
        label_col: Column name for the labels
        max_length: Maximum sequence length to consider
    """
    def __init__(self, dataset: Any, col_name: str = 'seqs', label_col: str = 'labels', max_length: int = 2048):
        self.seqs = dataset[col_name]
        self.labels = dataset[label_col]
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.seqs)
    
    def __getitem__(self, idx: int) -> Tuple[str, Union[float, int]]:
        seq = self.seqs[idx][:self.max_length]
        label = self.labels[idx]
        return seq, label


class PairCollator:
    """
    Collator for protein pair data that handles tokenization and tensor conversion.
    
    Args:
        tokenizer: The tokenizer to use for encoding sequences
        regression: Whether this is a regression task (True) or classification (False)
    """
    def __init__(self, tokenizer: Any, regression: bool = False):
        self.tokenizer = tokenizer
        self.regression = regression

    def __call__(self, batch: List[Tuple[str, str, Union[float, int]]]) -> Dict[str, torch.Tensor]:
        seqs_a, seqs_b, labels = zip(*batch)
        labels = torch.tensor(labels)
        if self.regression:
            labels = labels.float()
        else:
            labels = labels.long()
        tokenized = self.tokenizer(
            seqs_a, seqs_b,
            padding='longest',
            pad_to_multiple_of=8,
            return_tensors='pt'
        )
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels
        }


class SequenceCollator:
    """
    Collator for single protein sequence data that handles tokenization and tensor conversion.
    
    Args:
        tokenizer: The tokenizer to use for encoding sequences
        regression: Whether this is a regression task (True) or classification (False)
    """
    def __init__(self, tokenizer: Any, regression: bool = False):
        self.tokenizer = tokenizer
        self.regression = regression

    def __call__(self, batch: List[Tuple[str, Union[float, int]]]) -> Dict[str, torch.Tensor]:
        seqs, labels = zip(*batch)
        labels = torch.tensor(labels)
        if self.regression:
            labels = labels.float()
        else:
            labels = labels.long()
        tokenized = self.tokenizer(
            seqs,
            padding='longest',
            pad_to_multiple_of=8,
            return_tensors='pt'
        )
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels
        }


# Get the model ready, with or without LoRA
def initialize_model(model_name: str, num_labels: int, use_lora: bool = True, lora_config: Any = None):
    """
    Initialize a model with optional LoRA support
    
    Args:
        model_name: Name or path of the pretrained model
        num_labels: Number of labels for the task (1 for regression)
        use_lora: Whether to use LoRA for fine-tuning
        lora_config: Custom LoRA configuration (optional)
        
    Returns:
        model: The initialized model
        tokenizer: The model's tokenizer
    """
    print(f"Loading model {model_name} with {num_labels} labels...")
    
    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
        num_labels=num_labels
    )
    tokenizer = model.tokenizer
    
    # Apply LoRA if requested
    if use_lora:
        # Default LoRA configuration if none provided
        if lora_config is None:
            # Target modules for ESM++ or ESM2 models
            target_modules = ["layernorm_qkv.1", "out_proj", "query", "key", "value", "dense"]
            
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.01,
                bias="none",
                target_modules=target_modules,
            )
        
        # Apply LoRA to the model
        model = get_peft_model(model, lora_config)
        
        # Unfreeze the classifier head
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        # Print parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Non-trainable parameters: {non_trainable_params}")
        print(f"Percentage of parameters being trained: {100 * trainable_params / total_params:.2f}%")
    
    return model, tokenizer


# For computing performance metrics, it's fairly straightforward to add more metrics here
def compute_metrics_regression(p: EvalPrediction) -> Dict[str, float]:
    """Compute Spearman correlation for regression tasks"""
    predictions, labels = p.predictions, p.label_ids
    predictions = predictions[0] if isinstance(predictions, tuple) else predictions
    
    # Calculate Spearman correlation
    correlation, p_value = spearmanr(predictions.flatten(), labels.flatten())
    
    return {
        "spearman_correlation": correlation,
        "p_value": p_value
    }


def compute_metrics_classification(p: EvalPrediction) -> Dict[str, float]:
    """Compute accuracy for classification tasks"""
    predictions, labels = p.predictions, p.label_ids
    predictions = predictions[0] if isinstance(predictions, tuple) else predictions
    predictions = np.argmax(predictions, axis=-1)
    
    accuracy = (predictions.flatten() == labels.flatten()).mean()
    
    return {
        "accuracy": accuracy
    }


# For plotting the results, it's fairly straightforward to add more plots here
def plot_regression_results(preds: np.ndarray, labels: np.ndarray, task_name: str = "Regression") -> float:
    """
    Plot regression results with Spearman correlation
    
    Args:
        preds: Predicted values
        labels: True values
        task_name: Name of the task for plot title and filename
        
    Returns:
        correlation: Spearman correlation coefficient
    """
    # Calculate Spearman correlation
    correlation, p_value = spearmanr(preds, labels)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=labels, y=preds, alpha=0.6)
    
    # Add regression line
    sns.regplot(x=labels, y=preds, scatter=False, color='red')
    
    plt.title(f'{task_name} - Spearman Correlation: {correlation:.3f} (p={p_value:.3e})')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    
    # Add correlation text
    plt.annotate(f'ρ = {correlation:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', 
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{task_name.lower().replace(" ", "_")}_results.png')
    plt.show()
    return correlation


def plot_classification_results(trainer: Trainer, test_dataset: Any, task_name: str = "Classification") -> float:
    """
    Plot classification results with confusion matrix
    
    Args:
        trainer: The trained model trainer
        test_dataset: Dataset to evaluate on
        task_name: Name of the task for plot title and filename
        
    Returns:
        accuracy: Classification accuracy
    """
    # Get predictions
    predictions, labels, _ = trainer.predict(test_dataset)
    preds = predictions[0] if isinstance(predictions, tuple) else predictions
    pred_values = np.argmax(preds, axis=1)
    
    # Calculate accuracy
    accuracy = (pred_values == labels).mean()
    
    # Create confusion matrix
    cm = confusion_matrix(labels, pred_values)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    
    plt.title(f'{task_name} - Accuracy: {accuracy:.3f}')
    plt.tight_layout()
    plt.savefig(f'{task_name.lower().replace(" ", "_")}_results.png')
    plt.show()
    
    return accuracy


# Training functions
def train_regression_model(
        model_name: str = 'Synthyra/ESMplusplus_small',
        use_lora: bool = True,
        custom_lora_config: Any = None,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        num_epochs: int = 10,
        max_length: int = 1024,
        gradient_accumulation_steps: int = 1,
        patience: int = 3
    ) -> Tuple[Trainer, Any]:
    """
    Train a regression model for protein-protein affinity prediction
    
    Args:
        model_name: Name or path of the pretrained model
        use_lora: Whether to use LoRA for fine-tuning
        custom_lora_config: Custom LoRA configuration (optional)
        batch_size: Batch size for training
        learning_rate: Learning rate for training
        num_epochs: Number of epochs for training
        max_length: Maximum sequence length to consider
        gradient_accumulation_steps: Number of gradient accumulation steps
        patience: Number of evaluation calls with no improvement after which training will be stopped
        
    Returns:
        trainer: The trained model trainer
        test_dataset: The test dataset used for evaluation
    """
    print("Loading datasets for regression task...")
    
    # Filter sequences that exceed max_length
    def _filter_pair_by_length(example: Any) -> bool:
        return len(example['SeqA']) + len(example['SeqB']) <= max_length

    # Load datasets
    train_data = load_dataset('Synthyra/ProteinProteinAffinity', split='train').filter(_filter_pair_by_length)
    valid_data = load_dataset('Synthyra/AffinityBenchmarkv5.5', split='train').filter(_filter_pair_by_length)
    test_data = load_dataset('Synthyra/haddock_benchmark', split='train').filter(_filter_pair_by_length)

    # Create datasets
    train_dataset = PairDatasetHF(train_data, 'SeqA', 'SeqB', 'labels', max_length=max_length)
    valid_dataset = PairDatasetHF(valid_data, 'SeqA', 'SeqB', 'labels', max_length=max_length)
    test_dataset = PairDatasetHF(test_data, 'SeqA', 'SeqB', 'labels', max_length=max_length)
    
    # Initialize model with modular function
    model, tokenizer = initialize_model(
        model_name=model_name,
        num_labels=1,  # Regression task
        use_lora=use_lora,
        lora_config=custom_lora_config
    )
    
    # Create data collator
    data_collator = PairCollator(tokenizer, regression=True)
    
    # Define training arguments
    output_dir = "./results_regression_lora" if use_lora else "./results_regression"
    logging_dir = "./logs_regression_lora" if use_lora else "./logs_regression"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_dir=logging_dir,
        learning_rate=learning_rate,
        **BASE_TRAINER_KWARGS
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_regression,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)]
    )
    
    metrics = trainer.evaluate(test_dataset)
    print(f"Initial metrics: {metrics}")
    print("Training regression model...")
    trainer.train()
    
    # Evaluate and visualize results
    print("Evaluating and visualizing results...")
    predictions, labels, metrics = trainer.predict(test_dataset)
    preds = predictions[0] if isinstance(predictions, tuple) else predictions
    correlation = plot_regression_results(preds.flatten(), labels.flatten(), "Protein-Protein Affinity")
    print(f"Final Spearman correlation on test set: {correlation:.3f}")
    return trainer, test_dataset


def train_classification_model(
        model_name: str = 'Synthyra/ESMplusplus_small',
        use_lora: bool = True,
        custom_lora_config: Any = None,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        num_epochs: int = 10,
        max_length: int = 512,
        gradient_accumulation_steps: int = 1,
        patience: int = 3
    ) -> Tuple[Trainer, Any]:
    """
    Train a classification model for protein solubility prediction
    
    Args:
        model_name: Name or path of the pretrained model
        use_lora: Whether to use LoRA for fine-tuning
        custom_lora_config: Custom LoRA configuration (optional)
        batch_size: Batch size for training
        learning_rate: Learning rate for training
        num_epochs: Number of epochs for training
        max_length: Maximum sequence length to consider
        gradient_accumulation_steps: Number of gradient accumulation steps
        patience: Number of evaluation calls with no improvement after which training will be stopped
        
    Returns:
        trainer: The trained model trainer
    """
    print("Loading datasets for classification task...")
    
    # Filter sequences that exceed max_length
    def _filter_by_length(example: Any) -> bool:
        return len(example['seqs']) <= max_length

    # Load datasets
    data = load_dataset('GleghornLab/DL2_reg')
    train_data = data['train'].filter(_filter_by_length)
    valid_data = data['valid'].filter(_filter_by_length)
    test_data = data['test'].filter(_filter_by_length)

    # Create datasets
    train_dataset = SequenceDatasetHF(train_data, 'seqs', 'labels', max_length=max_length)
    valid_dataset = SequenceDatasetHF(valid_data, 'seqs', 'labels', max_length=max_length)
    test_dataset = SequenceDatasetHF(test_data, 'seqs', 'labels', max_length=max_length)
    
    # Get number of labels
    num_labels = len(set(train_data['labels']))
    
    # Initialize model with modular function
    model, tokenizer = initialize_model(
        model_name=model_name,
        num_labels=num_labels,
        use_lora=use_lora,
        lora_config=custom_lora_config
    )
    
    # Create data collator
    data_collator = SequenceCollator(tokenizer, regression=False)
    
    # Define training arguments
    output_dir = "./results_classification_lora" if use_lora else "./results_classification"
    logging_dir = "./logs_classification_lora" if use_lora else "./logs_classification"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_dir=logging_dir,
        learning_rate=learning_rate,
        **BASE_TRAINER_KWARGS
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_classification,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)]
    )
    
    metrics = trainer.evaluate(test_dataset)
    print(f"Initial metrics: {metrics}")
    print("Training classification model...")
    trainer.train()
    
    # Evaluate and visualize results
    print("Evaluating and visualizing results...")
    accuracy = plot_classification_results(trainer, test_dataset, "Protein Solubility")
    print(f"Final accuracy on test set: {accuracy:.3f}")
    
    return trainer


# Main function
if __name__ == "__main__":
    """
    With default arguments on 4070 laptop GPU
    py -m fine_tuning_example --task classification --batch_size 8 --epochs 2
        Runs in 80 seconds with test accuracy of ~89%
    py -m fine_tuning_example --task regression --batch_size 2 --max_length 1024 --grad_accum 4 --epochs 2
        Runs in 7 minutes with test Spearman correlation of ~0.72
    """
    import argparse
    
    # Examples of PLMs with efficient implementations offered by Synthyra
    MODEL_LIST = [
        'Synthyra/ESMplusplus_small',
        'Synthyra/ESMplusplus_large',
        'Synthyra/ESM2-8M',
        'Synthyra/ESM2-35M',
        'Synthyra/ESM2-150M',
        'Synthyra/ESM2-650M',
    ]

    parser = argparse.ArgumentParser(description="Train models for protein tasks")
    parser.add_argument("--task", type=str, choices=["regression", "classification", "both"], 
                        default="both", help="Task to train model for")
    parser.add_argument("--model_path", type=str, default="Synthyra/ESM2-8M",
                        help="Path to the model to train")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Whether to use LoRA for fine-tuning")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate for training")
    parser.add_argument("--epochs", type=float, default=1.0,
                        help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum length of input sequences")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience - number of evaluation calls with no improvement after which training will be stopped")
    args = parser.parse_args()

    # Print training configuration
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    print(f"Task: {args.task}")
    print(f"Using LoRA: {args.use_lora}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Number of epochs: {args.epochs}")
    print(f"Max sequence length: {args.max_length}")
    print(f"Gradient Accumulation Steps: {args.grad_accum}")
    print(f"Early stopping patience: {args.patience}")
    print("="*50 + "\n")
    
    # Train regression model if required
    if args.task in ["regression", "both"]:
        print("\n" + "="*50)
        print("TRAINING REGRESSION MODEL")
        print("="*50)
        regression_trainer, test_dataset = train_regression_model(
            model_name=args.model_path,
            use_lora=args.use_lora,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            max_length=args.max_length,
            gradient_accumulation_steps=args.grad_accum,
            patience=args.patience
        )

    # Train classification model if required
    if args.task in ["classification", "both"]:
        print("\n" + "="*50)
        print("TRAINING CLASSIFICATION MODEL")
        print("="*50)
        classification_trainer = train_classification_model(
            model_name=args.model_path,
            use_lora=args.use_lora,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            max_length=args.max_length,
            gradient_accumulation_steps=args.grad_accum,
            patience=args.patience
        )
    
    print("\nTraining completed!") 
