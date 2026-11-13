"""
Data loading and preparation utilities for LLaMA classification
Handles loading essays, rubrics, justifications and creating train/val/test splits
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data():
    """
    Load the ground truth essays and justifications.
    
    Returns:
        gt_df: DataFrame with essay texts and labels
        rubric_df: DataFrame with rubric information
        justification_df: DataFrame with justifications for each essay-subskill pair
    """
    gt_df = pd.read_excel("combined_essays.xlsx")
    gt_df.columns = gt_df.iloc[0]
    gt_df = gt_df[1:]
    
    rubric_df = pd.read_csv("rubric.csv")
    justification_df = pd.read_csv("justification_detailed_results.csv")
    
    logger.info(f"Loaded {len(gt_df)} essays, {len(rubric_df)} subskills, {len(justification_df)} justifications")
    
    return gt_df, rubric_df, justification_df


def create_data_splits(gt_df, random_state=47):
    """
    Create train/val/test splits EXACTLY as in modernbert_classify.py
    This ensures the same experimental setup.
    
    Returns: train_df, val_df, test_df (70/10/20 split)
    """
    shuffled_df = gt_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    train_df = shuffled_df.iloc[:int(0.7*len(shuffled_df))]
    val_df = shuffled_df.iloc[int(0.7*len(shuffled_df)):int(0.8*len(shuffled_df))]
    test_df = shuffled_df.iloc[int(0.8*len(shuffled_df)):]
    
    return train_df, val_df, test_df


def prepare_training_data(essay_df, rubric_df, justification_df, split_name):
    """
    Prepare combined dataset with essay text, labels, and justifications.
    
    Args:
        essay_df: DataFrame with essays (train/val/test split)
        rubric_df: DataFrame with rubric information
        justification_df: DataFrame with justifications
        split_name: 'train', 'val', or 'test' for logging
    
    Returns:
        combined_df: DataFrame with all necessary information for training
    """
    combined_data = []
    
    # Create a mapping from subskill to rubric info
    rubric_map = {}
    for _, rubric_row in rubric_df.iterrows():
        subskill_full = rubric_row['Subskill']  # e.g., "2.1: Synthesizing Multiple Sources"
        rubric_map[subskill_full] = {
            'definition': rubric_row['Definition'],
            'not_applicable': rubric_row['Not Applicable'],
            'below_emerging': rubric_row['Below Emerging'],
            'emerging': rubric_row['Emerging'],
            'expanding': rubric_row['Expanding'],
            'exemplifying': rubric_row['Exemplifying']
        }
        # Also create mapping with short form "Subskill 2.1" for matching with justification CSV
        subskill_short = f"Subskill {subskill_full.split(':')[0].strip()}"
        rubric_map[subskill_short] = rubric_map[subskill_full]
    
    # Create a lookup dictionary for justifications
    just_lookup = {}
    for _, just_row in justification_df.iterrows():
        essay_id = just_row['Essay ID']
        subskill_short = just_row['Subskill']  # e.g., "Subskill 2.1"
        key = (essay_id, subskill_short)
        just_lookup[key] = just_row['Justification']
    
    # Iterate through all essays in this split
    for _, essay_row in essay_df.iterrows():
        essay_id = str(essay_row['essay_id'])  # Ensure string type for consistency
        essay_text = essay_row['essay_text']
        
        # For each subskill in the rubric
        for subskill_full_key in rubric_map.keys():
            # Skip duplicate short-form keys (only process full format "X.X: Name")
            if ":" not in subskill_full_key:
                continue
            
            rubric_info = rubric_map[subskill_full_key]
            
            # Get the short form for looking up justification and label
            subskill_num = subskill_full_key.split(':')[0].strip()
            subskill_short = f"Subskill {subskill_num}"
            
            # Get label from essay_row (columns are named like "Subskill 2.1")
            if subskill_short not in essay_row:
                # Try the full form as fallback
                if subskill_full_key not in essay_row:
                    continue  # Skip if no label column exists
                label = essay_row[subskill_full_key]
            else:
                label = essay_row[subskill_short]
            
            # Skip if label is NaN or missing
            if pd.isna(label):
                continue
            
            # Get justification if available, otherwise use placeholder
            key = (essay_id, subskill_short)
            if key in just_lookup:
                justification = just_lookup[key]
            else:
                justification = "No detailed justification available for this essay-subskill assessment."
            
            combined_data.append({
                'essay_id': essay_id,
                'essay_text': essay_text,
                'subskill_full': subskill_full_key,
                'subskill_definition': rubric_info['definition'],
                'rubric_not_applicable': rubric_info['not_applicable'],
                'rubric_below_emerging': rubric_info['below_emerging'],
                'rubric_emerging': rubric_info['emerging'],
                'rubric_expanding': rubric_info['expanding'],
                'rubric_exemplifying': rubric_info['exemplifying'],
                'label': label,
                'justification': justification
            })
    
    combined_df = pd.DataFrame(combined_data)
    return combined_df


class LLaMADataset(Dataset):
    """PyTorch Dataset for LLaMA fine-tuning with generation"""
    
    def __init__(self, data_df, tokenizer, max_length=2048):
        """
        Args:
            data_df: DataFrame with columns: essay_id, essay_text, subskill_full, 
                     subskill_definition, label, justification
            tokenizer: HuggingFace tokenizer for LLaMA
            max_length: Maximum sequence length for the model
        """
        self.data = data_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Label mapping (numeric to text)
        self.label_map = {
            0: "not applicable",
            1: "below emerging",
            2: "emerging",
            3: "expanding",
            4: "exemplifying"
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Handle both numeric and text labels
        label_value = row['label']
        if isinstance(label_value, (int, np.integer)):
            label_text = self.label_map[label_value]
        else:
            label_text = str(label_value).lower()
        
        # Create user message (instruction + data)
        user_content = f"""Evaluate the essay for the given subskill. Provide a detailed justification explaining your assessment, then assign one of these proficiency labels: not applicable, below emerging, emerging, expanding, or exemplifying.

Subskill: {row['subskill_full']}
Definition: {row['subskill_definition']}

Essay:
{row['essay_text']}"""
        
        # Create assistant response
        assistant_content = f"""Justification: {row['justification']}
Label: {label_text}"""
        
        # Format as chat conversation using LLaMA's template
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
        
        # Apply chat template - this handles all special tokens automatically
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False  # Don't add prompt since we have the assistant response
        )
        
        # Tokenize the full conversation
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # For loss masking, tokenize just the user message to find where assistant response starts
        user_only_messages = [{"role": "user", "content": user_content}]
        user_text = self.tokenizer.apply_chat_template(
            user_only_messages,
            tokenize=False,
            add_generation_prompt=True  # Add the assistant prompt
        )
        user_encoding = self.tokenizer(
            user_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        input_length = user_encoding['input_ids'].shape[1]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'input_length': input_length,
            'essay_id': row['essay_id'],
            'subskill_full': row['subskill_full'],
            'label': row['label']
        }


def analyze_token_lengths(data_df, tokenizer, max_samples=None):
    """
    Analyze token lengths for the dataset
    
    Args:
        data_df: DataFrame with essay data
        tokenizer: HuggingFace tokenizer
        max_samples: Maximum number of samples to analyze (None = all)
    
    Returns:
        Dictionary with statistics
    """
    import matplotlib.pyplot as plt
    
    # Label can be either numeric or already text
    label_map = {
        0: "not applicable",
        1: "below emerging",
        2: "emerging",
        3: "expanding",
        4: "exemplifying"
    }
    
    input_lengths = []
    output_lengths = []
    total_lengths = []
    
    samples_to_analyze = len(data_df) if max_samples is None else min(max_samples, len(data_df))
    
    print(f"Analyzing token lengths for {samples_to_analyze} samples...")
    
    for idx in range(samples_to_analyze):
        row = data_df.iloc[idx]
        
        # Handle both numeric and text labels
        label_value = row['label']
        if isinstance(label_value, (int, np.integer)):
            label_text = label_map[label_value]
        else:
            label_text = str(label_value).lower()
        
        # User message
        user_content = f"""Evaluate the essay for the given subskill. Provide a detailed justification explaining your assessment, then assign one of these proficiency labels: not applicable, below emerging, emerging, expanding, or exemplifying.

Subskill: {row['subskill_full']}
Definition: {row['subskill_definition']}

Essay:
{row['essay_text']}"""
        
        # Assistant response
        assistant_content = f"""Justification: {row['justification']}
Label: {label_text}"""
        
        # Apply chat template
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        user_only_messages = [{"role": "user", "content": user_content}]
        input_text = tokenizer.apply_chat_template(user_only_messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize
        input_tokens = tokenizer(input_text, return_tensors='pt')
        output_tokens = tokenizer(assistant_content, return_tensors='pt')
        full_tokens = tokenizer(full_text, return_tensors='pt')
        
        input_len = input_tokens['input_ids'].shape[1]
        output_len = output_tokens['input_ids'].shape[1]
        total_len = full_tokens['input_ids'].shape[1]
        
        input_lengths.append(input_len)
        output_lengths.append(output_len)
        total_lengths.append(total_len)
    
    # Calculate statistics
    stats = {
        'input': {
            'min': np.min(input_lengths),
            'max': np.max(input_lengths),
            'mean': np.mean(input_lengths),
            'median': np.median(input_lengths),
            'p95': np.percentile(input_lengths, 95),
            'p99': np.percentile(input_lengths, 99),
        },
        'output': {
            'min': np.min(output_lengths),
            'max': np.max(output_lengths),
            'mean': np.mean(output_lengths),
            'median': np.median(output_lengths),
            'p95': np.percentile(output_lengths, 95),
            'p99': np.percentile(output_lengths, 99),
        },
        'total': {
            'min': np.min(total_lengths),
            'max': np.max(total_lengths),
            'mean': np.mean(total_lengths),
            'median': np.median(total_lengths),
            'p95': np.percentile(total_lengths, 95),
            'p99': np.percentile(total_lengths, 99),
        }
    }
    
    # Print statistics
    print("\n" + "="*70)
    print("TOKEN LENGTH STATISTICS")
    print("="*70)
    
    for section in ['input', 'output', 'total']:
        print(f"\n{section.upper()} (Input + Output):" if section == 'total' else f"\n{section.upper()}:")
        print(f"  Min:      {stats[section]['min']:6.0f} tokens")
        print(f"  Max:      {stats[section]['max']:6.0f} tokens")
        print(f"  Mean:     {stats[section]['mean']:6.1f} tokens")
        print(f"  Median:   {stats[section]['median']:6.0f} tokens")
        print(f"  95th %:   {stats[section]['p95']:6.0f} tokens")
        print(f"  99th %:   {stats[section]['p99']:6.0f} tokens")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    recommended_length = int(np.percentile(total_lengths, 99)) + 50  # Add buffer
    print(f"  Suggested max_length: {recommended_length} tokens")
    print(f"    (Covers 99% of samples with buffer)")
    print(f"  Samples exceeding 2048: {sum(1 for l in total_lengths if l > 2048)} ({sum(1 for l in total_lengths if l > 2048)/len(total_lengths)*100:.1f}%)")
    print(f"  Samples exceeding 1536: {sum(1 for l in total_lengths if l > 1536)} ({sum(1 for l in total_lengths if l > 1536)/len(total_lengths)*100:.1f}%)")
    print("="*70 + "\n")
    
    # Create distribution plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (section, lengths) in enumerate([('Input', input_lengths), 
                                               ('Output', output_lengths), 
                                               ('Total', total_lengths)]):
        axes[idx].hist(lengths, bins=50, edgecolor='black', alpha=0.7)
        axes[idx].axvline(stats[section.lower()]['mean'], color='red', 
                         linestyle='--', label=f'Mean: {stats[section.lower()]["mean"]:.0f}')
        axes[idx].axvline(stats[section.lower()]['median'], color='green', 
                         linestyle='--', label=f'Median: {stats[section.lower()]["median"]:.0f}')
        axes[idx].axvline(stats[section.lower()]['p95'], color='orange', 
                         linestyle='--', label=f'95th: {stats[section.lower()]["p95"]:.0f}')
        axes[idx].set_xlabel('Token Length')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{section} Token Distribution')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('token_length_analysis.png', dpi=150, bbox_inches='tight')
    print("📊 Token distribution plot saved to: token_length_analysis.png\n")
    plt.close()
    
    return stats, recommended_length


def create_formatting_func(tokenizer):
    """
    Create a formatting function for SFTTrainer.
    
    Args:
        tokenizer: HuggingFace tokenizer with chat template support
    
    Returns:
        Function that formats examples as chat-templated text
    """
    def formatting_func(example):
        """Format example with chat template for training."""
        # Handle both numeric and text labels
        label_value = example['label']
        if isinstance(label_value, (int, np.integer)):
            label_map = {
                0: "not applicable",
                1: "below emerging",
                2: "emerging",
                3: "expanding",
                4: "exemplifying"
            }
            label_text = label_map[label_value]
        else:
            label_text = str(label_value).lower()
        
        messages = [
            {
                "role": "user", 
                "content": f"""Evaluate the essay for the given subskill. Provide a detailed justification explaining your assessment, then assign one of these proficiency labels: not applicable, below emerging, emerging, expanding, or exemplifying.

Subskill: {example['subskill_full']}
Definition: {example['subskill_definition']}

Essay:
{example['essay_text']}"""
            },
            {
                "role": "assistant",
                "content": f"""Justification: {example['justification']}
Label: {label_text}"""
            }
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return text
    
    return formatting_func
