"""
ModernBERT classification for critical thinking essay evaluation
🎯 OPTIMIZED VERSION with best hyperparameters from tuning:
- Enhanced 2-layer classifier architecture  
- Learning rate: 2e-5
- Warmup ratio: 0.1
- Weight decay: 0.01
- Dropout: 0.1
- Expected accuracy: ~59.17% (vs 55.83% baseline)

Enhanced with multiple random seed training capability

We do not include rubric in the prompt.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import gc
import argparse
import wandb
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_gpu_memory():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class EnhancedModernBERTForSequenceClassification(nn.Module):
    """Enhanced classification head for ModernBERT with 2-layer classifier (BEST CONFIGURATION)"""
    def __init__(self, model_name="answerdotai/ModernBERT-base", num_labels=5, dropout_rate=0.1):
        super(EnhancedModernBERTForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.modernbert = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from model config
        self.hidden_size = self.modernbert.config.hidden_size
        
        # Enhanced classification head with layer normalization (from hyperparameter tuning)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Two-layer classifier for better representation
        self.intermediate = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_rate * 0.5)  # Less dropout in intermediate layer
        self.classifier = nn.Linear(self.hidden_size // 2, num_labels)
        
        # Initialize weights properly
        nn.init.normal_(self.intermediate.weight, std=0.02)
        nn.init.zeros_(self.intermediate.bias)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Get ModernBERT outputs
        outputs = self.modernbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use pooled output (CLS token representation)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        
        # Enhanced classification head
        pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout1(pooled_output)
        
        # Intermediate layer
        intermediate_output = self.intermediate(pooled_output)
        intermediate_output = self.activation(intermediate_output)
        intermediate_output = self.dropout2(intermediate_output)
        
        # Final classification
        logits = self.classifier(intermediate_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return type('', (), {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None
        })()

class ModernBERTForSequenceClassification(nn.Module):
    """Basic classification head for ModernBERT (kept for backward compatibility)"""
    def __init__(self, model_name="answerdotai/ModernBERT-base", num_labels=5, dropout_rate=0.1):
        super(ModernBERTForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.modernbert = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from model config
        self.hidden_size = self.modernbert.config.hidden_size
        
        # Custom classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        # Initialize classifier weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Get ModernBERT outputs
        outputs = self.modernbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use pooled output (CLS token representation)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return type('', (), {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None
        })()

class ModernBERTDataset:
    def __init__(self, input_ids, attention_masks, essay_ids, subskill_fulls, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.essay_ids = essay_ids
        self.subskill_fulls = subskill_fulls
        self.labels = labels
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'essay_id': self.essay_ids[idx],
            'subskill_full': self.subskill_fulls[idx],
            'label': self.labels[idx]
        }

def create_modernbert_dataloader(df, rubric_df, tokenizer, batch_size=16, datatype='train', max_length=1280):
    print(f'Creating ModernBERT {datatype} dataloader...')
    
    label_map = {
        "not applicable": 0, "n/a": 0, "na": 0,
        "below emerging": 1, "be": 1,
        "emerging": 2, "e": 2,
        "expanding": 3, "ex": 3,
        "exemplifying": 4, "exe": 4
    }
    
    all_input_ids = []
    all_attention_masks = []
    all_essay_ids = []
    all_subskill_fulls = []
    all_labels = []
    
    # Track tokenization statistics
    token_lengths = []
    original_lengths = []
    truncated_count = 0

    for index, data_row in df.iterrows():
        for _, rubric_row in rubric_df.iterrows():
            subskill_full = rubric_row['Subskill']
            subskill_col = f"Subskill {subskill_full.split(':')[0].strip()}"
            
            # Full rubric information - simplified to just definition as requested
            subskill_info = f"{subskill_full}\\nDefinition: {rubric_row['Definition']}"
            
            # Simplified input without the full rubric levels
            bert_input = f"Subskill Info: {subskill_info}\\nEssay: {data_row['essay_text']}"
            
            try:
                bert_label = label_map[data_row[subskill_col].lower().strip()]
            except:
                print(f"Warning: Unknown label {data_row[subskill_col]} for subskill {subskill_col}. Skipping.")
                continue
        
            # Tokenize with ModernBERT tokenizer
            encoding = tokenizer(bert_input, 
                                 padding='max_length', 
                                 truncation=True, 
                                 max_length=max_length, 
                                 return_tensors='pt')
            
            # Track statistics
            actual_length = (encoding['input_ids'] != tokenizer.pad_token_id).sum().item()
            token_lengths.append(actual_length)
            
            full_encoding = tokenizer(bert_input, return_tensors='pt')
            full_length = full_encoding['input_ids'].shape[1]
            original_lengths.append(full_length)
            if full_length > max_length:
                truncated_count += 1
            
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            
            essay_id = data_row['essay_id']
            label = torch.tensor(bert_label)

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_essay_ids.append(essay_id)
            all_subskill_fulls.append(subskill_full)
            all_labels.append(label)

    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_masks = torch.stack([item['attention_mask'] for item in batch])
        essay_ids = [item['essay_id'] for item in batch]
        subskill_fulls = [item['subskill_full'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        return input_ids, attention_masks, essay_ids, subskill_fulls, labels
    
    custom_dataset = ModernBERTDataset(all_input_ids, all_attention_masks, 
                                       all_essay_ids, all_subskill_fulls, all_labels)
    
    if datatype == 'train':  # Only print detailed stats for training dataset
        print(f"\\n=== ModernBERT Tokenization Statistics for {datatype} dataset ===")
        print(f"Total samples: {len(token_lengths)}")
        print(f"Samples truncated at {max_length} tokens: {truncated_count} ({truncated_count/len(token_lengths)*100:.1f}%)")
        print(f"Mean original length: {np.mean(original_lengths):.1f}")
        print(f"Mean preserved length: {np.mean(token_lengths):.1f}")
        print(f"Content preservation: {(1 - truncated_count/len(token_lengths))*100:.1f}%")
        print(f"Min original length: {min(original_lengths)}")
        print(f"Max original length: {max(original_lengths)}")
        print(f"95th percentile original: {np.percentile(original_lengths, 95):.1f}")
        print(f"99th percentile original: {np.percentile(original_lengths, 99):.1f}")
        print("=" * 60)
    
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader

def filter_dataloader_by_subskills(dataloader, target_subskills):
    """
    Filter a dataloader to only include specific subskills
    
    Args:
        dataloader: The original dataloader
        target_subskills: List of subskill numbers to include (e.g., [2, 3] for subskills 2.1, 2.2, 3.1, 3.2)
    
    Returns:
        Filtered dataloader containing only the target subskills
    """
    filtered_input_ids = []
    filtered_attention_masks = []
    filtered_essay_ids = []
    filtered_subskill_fulls = []
    filtered_labels = []
    
    # Get all data from the original dataset
    dataset = dataloader.dataset
    
    for i in range(len(dataset)):
        item = dataset[i]
        subskill_full = item['subskill_full']
        
        # Extract subskill number (e.g., "2.1: ..." -> 2)
        subskill_num = int(subskill_full.split('.')[0])
        
        if subskill_num in target_subskills:
            filtered_input_ids.append(item['input_ids'])
            filtered_attention_masks.append(item['attention_mask'])
            filtered_essay_ids.append(item['essay_id'])
            filtered_subskill_fulls.append(item['subskill_full'])
            filtered_labels.append(item['label'])
    
    if not filtered_input_ids:
        print(f"Warning: No data found for subskills {target_subskills}")
        return None
    
    # Create new dataset with filtered data
    filtered_dataset = ModernBERTDataset(
        filtered_input_ids, filtered_attention_masks,
        filtered_essay_ids, filtered_subskill_fulls, filtered_labels
    )
    
    # Create new dataloader with same batch size and collate function
    filtered_dataloader = DataLoader(
        filtered_dataset, 
        batch_size=dataloader.batch_size, 
        shuffle=True, 
        collate_fn=dataloader.collate_fn
    )
    
    print(f"Filtered dataloader: {len(filtered_dataset)} samples for subskills {target_subskills}")
    return filtered_dataloader

def create_subskill_split_dataloader(df, rubric_df, tokenizer, train_subskills, test_subskills, split_type='train', val_ratio=0.1, batch_size=16, max_length=1280, random_state=47):
    """
    Create dataloaders based on subskill splits rather than essay splits
    
    Args:
        df: Full dataframe with all essays
        rubric_df: Full rubric dataframe  
        tokenizer: ModernBERT tokenizer
        train_subskills: List of subskill numbers for training/validation (e.g., [2, 3])
        test_subskills: List of subskill numbers for testing (e.g., [4])
        split_type: 'train', 'val', or 'test'
        val_ratio: Ratio of training subskills to use for validation (default 0.1 = 10%)
        batch_size: Batch size for dataloader
        max_length: Max sequence length
        random_state: Random state for train/val split
    
    Returns:
        DataLoader for the specified split
    """
    
    if split_type == 'test':
        # For test, use all essays but only test subskills
        filtered_rubric = rubric_df[rubric_df['Subskill'].str.startswith(tuple([f"{num}." for num in test_subskills]))]
        print(f"Creating TEST dataloader with subskills: {list(filtered_rubric['Subskill'])}")
        return create_modernbert_dataloader(df, filtered_rubric, tokenizer, batch_size, 'test', max_length)
    
    else:
        # For train/val, use all essays but only training subskills
        filtered_rubric = rubric_df[rubric_df['Subskill'].str.startswith(tuple([f"{num}." for num in train_subskills]))]
        
        # Create full dataloader with training subskills
        full_dataloader = create_modernbert_dataloader(df, filtered_rubric, tokenizer, batch_size, 'temp', max_length)
        
        # Extract all data from the dataloader
        all_data = []
        for batch in full_dataloader:
            input_ids, attention_masks, essay_ids, subskill_fulls, labels = batch
            for i in range(len(input_ids)):
                all_data.append({
                    'input_ids': input_ids[i],
                    'attention_mask': attention_masks[i], 
                    'essay_id': essay_ids[i],
                    'subskill_full': subskill_fulls[i],
                    'label': labels[i]
                })
        
        # Shuffle the data with fixed random state
        np.random.seed(random_state)
        np.random.shuffle(all_data)
        
        # Split into train and val based on val_ratio
        val_size = int(len(all_data) * val_ratio)
        
        if split_type == 'train':
            split_data = all_data[val_size:]  # Use remaining for training
            print(f"Creating TRAIN dataloader with {len(split_data)} samples from subskills: {list(filtered_rubric['Subskill'])}")
        elif split_type == 'val':
            split_data = all_data[:val_size]  # Use first val_size for validation
            print(f"Creating VAL dataloader with {len(split_data)} samples from subskills: {list(filtered_rubric['Subskill'])}")
        
        # Create custom dataset from split data
        class ModernBERTDataset:
            def __init__(self, data):
                self.data = data
                
            def __len__(self):
                return len(self.data)
                
            def __getitem__(self, idx):
                return self.data[idx]
        
        def collate_fn(batch):
            input_ids = torch.stack([item['input_ids'] for item in batch])
            attention_masks = torch.stack([item['attention_mask'] for item in batch])
            essay_ids = [item['essay_id'] for item in batch]
            subskill_fulls = [item['subskill_full'] for item in batch]
            labels = torch.stack([item['label'] for item in batch])
            return input_ids, attention_masks, essay_ids, subskill_fulls, labels
        
        split_dataset = ModernBERTDataset(split_data)
        return DataLoader(split_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

def perform_modernbert_train(model, tokenizer, train_loader, val_loader, device, 
                            num_epochs=6, lr=2e-5, results_dir="modernbert_results", 
                            use_wandb=False):
    print("### ModernBERT Training Started ###")
    
    # Clear GPU cache before starting
    torch.cuda.empty_cache()
    gc.collect()
    check_gpu_memory()
    
    # Calculate class weights for imbalanced dataset
    all_labels = []
    temp_loader = DataLoader(train_loader.dataset, batch_size=len(train_loader.dataset), 
                            shuffle=False, collate_fn=train_loader.collate_fn)
    for batch in temp_loader:
        _, _, _, _, labels = batch
        all_labels.extend(labels.numpy())
        break
    
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = np.sqrt(class_weights)  # Apply smoothing to reduce extremes
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(f"Smoothed class weights: {class_weights}")
    
    # Training hyperparameters - matching Longformer exactly
    warmup_ratio = 0.1  # Match Longformer's 0.1 warmup
    patience = 3  # Match Longformer's patience
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
    
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * total_steps),
        num_training_steps=total_steps
    )
    
    best_val_accuracy = 0.0
    patience_counter = 0
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Training config: lr={lr}, epochs={num_epochs}, warmup={warmup_ratio}, patience={patience}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            input_ids, attention_masks, essay_ids, subskill_fulls, labels = batch
            
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through ModernBERT
            outputs = model(input_ids=input_ids, 
                           attention_mask=attention_masks,
                           labels=labels)
            
            # Use focal loss for better class imbalance handling
            focal_loss_fn = FocalLoss(alpha=1, gamma=2, weight=class_weights_tensor)
            loss = focal_loss_fn(outputs.logits, labels)
            
            total_loss += loss.item()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}, Average Training Loss: {avg_train_loss:.4f}, LR: {current_lr:.2e}")
        
        if use_wandb:
            wandb.log({
                "train_loss": avg_train_loss,
                "learning_rate": current_lr,
                "epoch": epoch + 1
            }, step=epoch+1)

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}"):
                input_ids, attention_masks, essay_ids, subskill_fulls, labels = batch
                
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                labels = labels.to(device)
                
                outputs = model(input_ids=input_ids,
                               attention_mask=attention_masks,
                               labels=labels)
                
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                val_loss += outputs.loss.item()
        
        val_accuracy = correct / total
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}")

        if use_wandb:
            wandb.log({
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy
            }, step=epoch+1)

        # Early stopping and model saving
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            
            # Save the best model
            model_save_path = os.path.join(results_dir, "best_model")
            os.makedirs(model_save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_save_path, "pytorch_model.bin"))
            tokenizer.save_pretrained(model_save_path)
            
            # Save model config for reconstruction
            config = {
                "model_name": "answerdotai/ModernBERT-base",
                "num_labels": 5,
                "dropout_rate": 0.1,
                "max_length": 1280,
                "best_accuracy": best_val_accuracy
            }
            with open(os.path.join(model_save_path, "config.json"), "w") as f:
                json.dump(config, f, indent=2)
            
            print(f"Best model saved with accuracy: {best_val_accuracy:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Memory cleanup after each epoch
        torch.cuda.empty_cache()
        gc.collect()

    print("### ModernBERT Training Complete ###")
    return best_val_accuracy

def perform_modernbert_inference(model, test_loader, device, results_dir="modernbert_results", load_best_model=True):
    print("### ModernBERT Inference Started ###")

    # Load the best saved model if requested
    if load_best_model:
        best_model_path = os.path.join(results_dir, "best_model")
        if os.path.exists(best_model_path):
            print(f"Loading best model from: {best_model_path}")
            
            # Load the config to determine which architecture to use
            config_path = os.path.join(best_model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                dropout_rate = config.get('dropout_rate', 0.1)
            else:
                dropout_rate = 0.1  # Default
            
            # Create new model instance and load weights
            model = EnhancedModernBERTForSequenceClassification(
                model_name="answerdotai/ModernBERT-base",
                num_labels=5,
                dropout_rate=dropout_rate
            )
            
            # Load the saved state dict
            state_dict_path = os.path.join(best_model_path, "pytorch_model.bin")
            model.load_state_dict(torch.load(state_dict_path, map_location=device))
            model.to(device)
            
            print(f"✅ Best model loaded successfully!")
        else:
            print(f"⚠️  Best model not found at {best_model_path}, using current model")

    reverse_label_map = {
        0: "not applicable",
        1: "below emerging", 
        2: "emerging",
        3: "expanding",
        4: "exemplifying"
    }

    model.eval()
    test_results = defaultdict(dict)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids, attention_masks, essay_ids, subskill_fulls, labels = batch
            
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_masks)
            predictions = torch.argmax(outputs.logits, dim=1)
            
            # Convert to numpy for easier handling
            pred_labels = predictions.cpu().numpy()
            
            for i in range(len(essay_ids)):
                essay_id = essay_ids[i]
                subskill = subskill_fulls[i]
                pred_label = pred_labels[i]
                
                pred_text = reverse_label_map[pred_label]
                
                test_results[essay_id][subskill] = {
                    'label': pred_text
                }

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    
    # Save test results in same format as other models
    with open(os.path.join(results_dir, "test_results.json"), "w") as f:
        json.dump(dict(test_results), f, indent=4)
    
    print(f"Results saved to {os.path.join(results_dir, 'test_results.json')}")
    print("### ModernBERT Inference Complete ###")

def create_data_splits(gt_df, random_state=47):
    """
    Create train/val/test splits for a given random state
    Returns: train_df, val_df, test_df
    """
    # shuffle the dataframe with specific random state
    shuffled_df = gt_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # split into train, val, and test (70, 10, 20)
    train_df = shuffled_df.iloc[:int(0.7*len(shuffled_df))]
    val_df = shuffled_df.iloc[int(0.7*len(shuffled_df)):int(0.8*len(shuffled_df))]
    test_df = shuffled_df.iloc[int(0.8*len(shuffled_df)):]

    # log train/val/test sizes
    logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)} for random_state={random_state}")
    
    return train_df, val_df, test_df

def train_single_model(train_df, val_df, test_df, rubric_df, device, random_state, results_dir, use_wandb=False):
    """
    Train a single ModernBERT model for a specific random state split
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training ModernBERT model for random_state={random_state}")
    logger.info(f"Results will be saved to: {results_dir}")
    logger.info(f"{'='*60}")
    
    # Create fresh model and tokenizer for each split
    model_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize ENHANCED classification model (best from hyperparameter tuning)
    model = EnhancedModernBERTForSequenceClassification(
        model_name=model_name, 
        num_labels=5, 
        dropout_rate=0.1  # Optimal dropout rate
    )
    model.to(device)
    
    # Create dataloaders
    print('## Creating ModernBERT Dataloaders ##')
    MAX_LENGTH = 1280  # Reduced based on prompt length analysis
    print(f"Using max_length = {MAX_LENGTH} tokens (3x BERT capacity)")
    
    train_loader = create_modernbert_dataloader(train_df, rubric_df, tokenizer, batch_size=16, datatype='train', max_length=MAX_LENGTH)
    val_loader = create_modernbert_dataloader(val_df, rubric_df, tokenizer, batch_size=16, datatype='val', max_length=MAX_LENGTH)
    test_loader = create_modernbert_dataloader(test_df, rubric_df, tokenizer, batch_size=16, datatype='test', max_length=MAX_LENGTH)
    
    # Train the model with OPTIMAL hyperparameters
    print(f"## Training ModernBERT Model (random_state={random_state}) ##")
    perform_modernbert_train(model, tokenizer, train_loader, val_loader, device, 
                            num_epochs=6, lr=2e-5, results_dir=results_dir, use_wandb=use_wandb)
    
    # Inference on test set (will load best model automatically)
    print(f"## Running ModernBERT Inference (random_state={random_state}) ##")
    perform_modernbert_inference(None, test_loader, device, results_dir=results_dir, load_best_model=True)
    
    # Clean up to free memory
    del model, tokenizer, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    
    print(f"Completed ModernBERT training for random_state={random_state}")

def train_multiple_splits(gt_df, rubric_df, device, base_results_dir="modernbert_results_splits", use_wandb=False, wandb_project=None, wandb_base_name=None):
    """
    Train ModernBERT models for multiple random states (splits)
    """
    # Define the random states to use (same as BERT for consistency)
    random_states = [47, 123, 456, 789, 999]
    
    print(f"\n{'='*80}")
    print(f"TRAINING MODERNBERT MODELS WITH MULTIPLE RANDOM SPLITS")
    print(f"Random states: {random_states}")
    print(f"Base results directory: {base_results_dir}")
    print(f"Using OPTIMAL hyperparameters from tuning:")
    print(f"   - Architecture: Enhanced (2-layer classifier)")
    print(f"   - Learning rate: 2e-5")
    print(f"   - Warmup ratio: 0.1")
    print(f"   - Weight decay: 0.01")
    print(f"   - Dropout: 0.1")
    print(f"   - Expected accuracy: ~59.17%")
    print(f"{'='*80}")
    
    if wandb_base_name is None:
        wandb_base_name = "modernbert_splits"

    for i, random_state in enumerate(random_states):
        print(f"\n[{i+1}/{len(random_states)}] Processing random_state={random_state}")
        
        # Initialize wandb for this split if requested
        if use_wandb:
            run_name = f"{wandb_base_name}_seed_{random_state}"
            wandb.init(project=wandb_project, name=run_name, 
                       config={"random_state": random_state, "experiment_type": "multiple_splits"},
                       reinit=True)

        # Create data splits for this random state
        train_df, val_df, test_df = create_data_splits(gt_df, random_state)
        
        # Create results directory for this split
        split_results_dir = os.path.join(base_results_dir, f"rstate_{random_state}")
        
        # Train model for this split
        train_single_model(train_df, val_df, test_df, rubric_df, device, random_state, split_results_dir, use_wandb=use_wandb)
        
        if use_wandb:
            wandb.finish()
            
        print(f"✅ Completed ModernBERT split {i+1}/{len(random_states)} (random_state={random_state})")
    
    print(f"\n{'='*80}")
    print(f"ALL MODERNBERT SPLITS COMPLETED!")
    print(f"Results saved in: {base_results_dir}/")
    print(f"Split directories: {[f'rstate_{rs}' for rs in random_states]}")
    print(f"{'='*80}")

def run_generalization_experiment(gt_df, rubric_df, device, base_results_dir="modernbert_results_generalize", use_wandb=False, wandb_project=None, wandb_base_name=None):
    """
    Run generalization experiment: train on 2 subskills, test on the third
    
    Experiments:
    1. Train on subskills 2,3 -> Test on subskill 4
    2. Train on subskills 2,4 -> Test on subskill 3  
    3. Train on subskills 3,4 -> Test on subskill 2
    """
    print(f"\n{'='*80}")
    print(f"MODERNBERT GENERALIZATION EXPERIMENT")
    print(f"Training on 2 subskills, testing on the third")
    print(f"Base results directory: {base_results_dir}")
    print(f"{'='*80}")
    
    if wandb_base_name is None:
        wandb_base_name = "modernbert_generalize"

    # Define the three experiments
    experiments = [
        {
            'train_subskills': [2, 3],
            'test_subskills': [4],
            'name': 'subskill_4',
            'description': 'Train on subskills 2,3 -> Test on subskill 4'
        },
        {
            'train_subskills': [2, 4], 
            'test_subskills': [3],
            'name': 'subskill_3',
            'description': 'Train on subskills 2,4 -> Test on subskill 3'
        },
        {
            'train_subskills': [3, 4],
            'test_subskills': [2], 
            'name': 'subskill_2',
            'description': 'Train on subskills 3,4 -> Test on subskill 2'
        }
    ]
    
    # Create 70-10-20 split with random_state=47
    print("Creating 70-10-20 data split with random_state=47")
    train_df, val_df, test_df = create_data_splits(gt_df, random_state=47)
    
    for i, experiment in enumerate(experiments):
        print(f"\n[{i+1}/3] {experiment['description']}")
        print("-" * 60)
        
        # Initialize wandb for this experiment if requested
        if use_wandb:
            run_name = f"{wandb_base_name}_{experiment['name']}"
            wandb.init(project=wandb_project, name=run_name, 
                       config={"experiment": experiment, "type": "generalization"},
                       reinit=True)
                       
        # Create results directory for this experiment
        exp_results_dir = os.path.join(base_results_dir, experiment['name'])
        os.makedirs(exp_results_dir, exist_ok=True)
        
        # Initialize fresh model and tokenizer
        model_name = "answerdotai/ModernBERT-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = EnhancedModernBERTForSequenceClassification(
            model_name=model_name,
            num_labels=5,
            dropout_rate=0.1
        )
        model.to(device)
        
        # Create full dataloaders first
        print("Creating full dataloaders...")
        MAX_LENGTH = 1280
        
        full_train_loader = create_modernbert_dataloader(train_df, rubric_df, tokenizer, 
                                                        batch_size=16, datatype='train', max_length=MAX_LENGTH)
        full_val_loader = create_modernbert_dataloader(val_df, rubric_df, tokenizer,
                                                      batch_size=16, datatype='val', max_length=MAX_LENGTH)
        full_test_loader = create_modernbert_dataloader(test_df, rubric_df, tokenizer,
                                                       batch_size=16, datatype='test', max_length=MAX_LENGTH)
        
        # Filter dataloaders for training/validation (use train subskills)
        print(f"Filtering train/val data for subskills: {experiment['train_subskills']}")
        filtered_train_loader = filter_dataloader_by_subskills(full_train_loader, experiment['train_subskills'])
        filtered_val_loader = filter_dataloader_by_subskills(full_val_loader, experiment['train_subskills'])
        
        # Filter test dataloader (use test subskills)
        print(f"Filtering test data for subskills: {experiment['test_subskills']}")
        filtered_test_loader = filter_dataloader_by_subskills(full_test_loader, experiment['test_subskills'])
        
        # Check if we have data for all loaders
        if filtered_train_loader is None or filtered_val_loader is None or filtered_test_loader is None:
            print(f"❌ Skipping experiment due to missing data")
            continue
        
        # Train the model
        print(f"Training ModernBERT model for {experiment['name']}")
        perform_modernbert_train(model, tokenizer, filtered_train_loader, filtered_val_loader, 
                                device, num_epochs=6, lr=2e-5, results_dir=exp_results_dir, use_wandb=use_wandb)
        
        # Test the model
        print(f"Testing ModernBERT model on {experiment['test_subskills']}")
        perform_modernbert_inference(None, filtered_test_loader, device, 
                                    results_dir=exp_results_dir, load_best_model=True)
        
        if use_wandb:
            wandb.finish()
        
        # Save experiment info
        exp_info = {
            'experiment_name': experiment['name'],
            'description': experiment['description'],
            'train_subskills': experiment['train_subskills'],
            'test_subskills': experiment['test_subskills'],
            'train_samples': len(filtered_train_loader.dataset),
            'val_samples': len(filtered_val_loader.dataset),
            'test_samples': len(filtered_test_loader.dataset)
        }
        
        with open(os.path.join(exp_results_dir, "experiment_info.json"), "w") as f:
            json.dump(exp_info, f, indent=2)
        
        # Clean up memory
        del model, tokenizer, full_train_loader, full_val_loader, full_test_loader
        del filtered_train_loader, filtered_val_loader, filtered_test_loader
        torch.cuda.empty_cache()
        
        print(f"✅ Completed experiment {i+1}/3: {experiment['name']}")
    
    print(f"\n{'='*80}")
    print(f"ALL MODERNBERT GENERALIZATION EXPERIMENTS COMPLETED!")
    print(f"Results saved in: {base_results_dir}/")
    print(f"Experiment directories: {[exp['name'] for exp in experiments]}")
    print(f"{'='*80}")

def run_subskill_finetuning_experiments(gt_df, rubric_df, device, base_results_dir="modernbert_results_gen_subskills", use_wandb=False, wandb_project=None, wandb_base_name=None):
    """
    Run subskill-based fine-tuning experiments: split by subskills, not essays
    
    For each experiment:
    - Training subskills are split 90% train, 10% validation (across all essays)
    - Test subskill uses all essays for testing
    
    Experiments:
    1. Train on subskills 2,3 (90% train, 10% val) -> Test on subskill 4 (all essays)
    2. Train on subskills 2,4 (90% train, 10% val) -> Test on subskill 3 (all essays)  
    3. Train on subskills 3,4 (90% train, 10% val) -> Test on subskill 2 (all essays)
    """
    # Define the experiments
    experiments = [
        {"train_subskills": [2, 3], "test_subskill": 4, "name": "subskill_4"},
        {"train_subskills": [2, 4], "test_subskill": 3, "name": "subskill_3"},
        {"train_subskills": [3, 4], "test_subskill": 2, "name": "subskill_2"}
    ]
    
    print(f"\n{'='*80}")
    print(f"RUNNING MODERNBERT SUBSKILL-BASED FINE-TUNING EXPERIMENTS")
    print(f"Setup: Split by subskills (not essays)")
    print(f"Training subskills: 90% train, 10% val across all essays")
    print(f"Test subskill: 100% of essays for testing")
    print(f"Base results directory: {base_results_dir}")
    print(f"{'='*80}")

    if wandb_base_name is None:
        wandb_base_name = "modernbert"
    
    for i, exp in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] Experiment: Train on subskills {exp['train_subskills']}, Test on subskill {exp['test_subskill']}")
        
        if use_wandb:
            run_name = f"{wandb_base_name}_{exp['name']}"
            wandb.init(project=wandb_project, name=run_name, 
                       config={"experiment": exp, "type": "subskill_finetuning"},
                       reinit=True)

        # Create results directory for this experiment
        exp_results_dir = os.path.join(base_results_dir, exp['name'])
        os.makedirs(exp_results_dir, exist_ok=True)
        
        # Initialize model and tokenizer for this experiment
        model_name = "answerdotai/ModernBERT-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = EnhancedModernBERTForSequenceClassification(
            model_name=model_name, 
            num_labels=5, 
            dropout_rate=0.1
        )
        model = model.to(device)
        
        # Create subskill-split dataloaders
        print(f"Creating subskill-based dataloaders...")
        MAX_LENGTH = 1280
        VAL_RATIO = 0.1  # 10% for validation, 90% for training
        
        # Train dataloader: 90% of training subskills across all essays
        train_loader = create_subskill_split_dataloader(
            gt_df, rubric_df, tokenizer, 
            train_subskills=exp['train_subskills'], 
            test_subskills=[exp['test_subskill']],
            split_type='train', 
            val_ratio=VAL_RATIO,
            batch_size=16, 
            max_length=MAX_LENGTH,
            random_state=47
        )
        
        # Val dataloader: 10% of training subskills across all essays  
        val_loader = create_subskill_split_dataloader(
            gt_df, rubric_df, tokenizer,
            train_subskills=exp['train_subskills'],
            test_subskills=[exp['test_subskill']],
            split_type='val',
            val_ratio=VAL_RATIO, 
            batch_size=16,
            max_length=MAX_LENGTH,
            random_state=47
        )
        
        # Test dataloader: 100% of test subskill across all essays
        test_loader = create_subskill_split_dataloader(
            gt_df, rubric_df, tokenizer,
            train_subskills=exp['train_subskills'],
            test_subskills=[exp['test_subskill']],
            split_type='test',
            batch_size=16,
            max_length=MAX_LENGTH
        )
        
        print(f"Training on subskills {exp['train_subskills']}: {len(train_loader.dataset)} train samples, {len(val_loader.dataset)} val samples")
        print(f"Testing on subskill {exp['test_subskill']}: {len(test_loader.dataset)} test samples")
        
        # Save experiment info
        exp_info = {
            'experiment_name': exp['name'],
            'train_subskills': exp['train_subskills'],
            'test_subskill': exp['test_subskill'],
            'split_method': 'subskill_based',
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'test_samples': len(test_loader.dataset),
            'val_ratio': VAL_RATIO,
            'description': f"Train on subskills {exp['train_subskills']} (90% train, 10% val), Test on subskill {exp['test_subskill']} (100% test)"
        }
        
        with open(os.path.join(exp_results_dir, "experiment_info.json"), "w") as f:
            json.dump(exp_info, f, indent=2)
        
        # Train the model
        perform_modernbert_train(model, tokenizer, train_loader, val_loader, device, 
                               num_epochs=6, lr=2e-5, results_dir=exp_results_dir, use_wandb=use_wandb)
        
        # Load best model and test
        print(f"Loading best model for testing...")
        best_model_path = os.path.join(exp_results_dir, "best_model")
        model = EnhancedModernBERTForSequenceClassification(
            model_name=model_name, 
            num_labels=5, 
            dropout_rate=0.1
        )
        model.load_state_dict(torch.load(os.path.join(best_model_path, "pytorch_model.bin")))
        model = model.to(device)
        
        # Test the model
        perform_modernbert_inference(model, test_loader, device, results_dir=exp_results_dir)
        
        # Clean up memory
        del model, tokenizer, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()
        
        if use_wandb:
            wandb.finish()

        print(f"✅ Completed experiment {i+1}/{len(experiments)}: {exp['name']}")
    
    print(f"\n{'='*80}")
    print(f"ALL MODERNBERT SUBSKILL FINE-TUNING EXPERIMENTS COMPLETED!")
    print(f"Results saved in: {base_results_dir}/")
    print(f"Experiment directories: {[exp['name'] for exp in experiments]}")
    print(f"{'='*80}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ModernBERT Classification with optional multiple splits')
    parser.add_argument('--multiple_splits', action='store_true', 
                       help='Train 5 different ModernBERT models with different random states (47, 123, 456, 789, 999)')
    parser.add_argument('--generalize', action='store_true',
                       help='Run generalization experiment: train on 2 subskills, test on the third')
    parser.add_argument('--gen_subskills', action='store_true',
                       help='Run subskill-based fine-tuning: split by subskills, not essays')
    parser.add_argument('--results_dir', type=str, default='modernbert_results',
                       help='Results directory (default: modernbert_results, or modernbert_results_splits for multiple splits)')
    parser.add_argument('--use_wandb', action='store_true', help='Log training metrics to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default="critical-thinking", help='WandB project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='WandB run name')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    print("### ModernBERT Classification Started ###")

    # Load data
    rubric_df = pd.read_csv("rubric.csv")
    gt_df = pd.read_excel("combined_essays.xlsx")
    gt_df.columns = gt_df.iloc[0]
    gt_df = gt_df[1:]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.multiple_splits:
        # Train multiple models with different random states
        base_results_dir = "modernbert_results_splits" if args.results_dir == "modernbert_results" else args.results_dir
        train_multiple_splits(gt_df, rubric_df, device, base_results_dir, 
                             use_wandb=args.use_wandb, wandb_project=args.wandb_project, wandb_base_name=args.wandb_name)
    elif args.generalize:
        # Run generalization experiment
        base_results_dir = "modernbert_results_generalize" if args.results_dir == "modernbert_results" else args.results_dir
        run_generalization_experiment(gt_df, rubric_df, device, base_results_dir, 
                                     use_wandb=args.use_wandb, wandb_project=args.wandb_project, wandb_base_name=args.wandb_name)
    elif args.gen_subskills:
        # Run subskill-based fine-tuning experiments
        base_results_dir = "modernbert_results_gen_subskills" if args.results_dir == "modernbert_results" else args.results_dir
        run_subskill_finetuning_experiments(gt_df, rubric_df, device, base_results_dir, 
                                           use_wandb=args.use_wandb, wandb_project=args.wandb_project, wandb_base_name=args.wandb_name)
    else:
        # Original single model training
        print("Training single ModernBERT model with original functionality")
        
        if args.use_wandb:
            run_name = args.wandb_name if args.wandb_name else "modernbert_single"
            wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        
        # Create data splits with original random state (47)
        gt_df = gt_df.sample(frac=1, random_state=47).reset_index(drop=True)

        # Split data
        train_df = gt_df.iloc[:int(0.7*len(gt_df))]
        val_df = gt_df.iloc[int(0.7*len(gt_df)):int(0.8*len(gt_df))]
        test_df = gt_df.iloc[int(0.8*len(gt_df)):]
        
        # Initialize ModernBERT tokenizer and model
        print("Loading ModernBERT model with OPTIMAL HYPERPARAMETERS...")
        print("🎯 Using configuration from hyperparameter tuning:")
        print("   - Architecture: Enhanced (2-layer classifier)")
        print("   - Learning rate: 2e-5")
        print("   - Warmup ratio: 0.1")
        print("   - Weight decay: 0.01")
        print("   - Dropout: 0.1")
        print("   - Expected accuracy: ~59.17%")
        
        model_name = "answerdotai/ModernBERT-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special tokens if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Initialize ENHANCED classification model (best from hyperparameter tuning)
        model = EnhancedModernBERTForSequenceClassification(
            model_name=model_name, 
            num_labels=5, 
            dropout_rate=0.1  # Optimal dropout rate
        )
        model.to(device)

        print('## Creating ModernBERT Dataloaders ##')
        MAX_LENGTH = 1280  # Reduced based on prompt length analysis
        print(f"Using max_length = {MAX_LENGTH} tokens (3x BERT capacity)")
        
        train_loader = create_modernbert_dataloader(train_df, rubric_df, tokenizer, batch_size=16, datatype='train', max_length=MAX_LENGTH)
        val_loader = create_modernbert_dataloader(val_df, rubric_df, tokenizer, batch_size=16, datatype='val', max_length=MAX_LENGTH)
        test_loader = create_modernbert_dataloader(test_df, rubric_df, tokenizer, batch_size=16, datatype='test', max_length=MAX_LENGTH)
        
        # Train the model with OPTIMAL hyperparameters
        perform_modernbert_train(model, tokenizer, train_loader, val_loader, device, 
                                num_epochs=6, lr=2e-5, results_dir=args.results_dir, use_wandb=args.use_wandb)

        # Test the model
        perform_modernbert_inference(model, test_loader, device, results_dir=args.results_dir)

if __name__ == "__main__":
    main()
