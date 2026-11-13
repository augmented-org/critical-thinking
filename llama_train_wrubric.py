"""
Training script for fine-tuning Llama 3.1 8B for critical thinking essay evaluation.
Uses LoRA + 4-bit quantization for efficient training.

Usage:
    python llama_train.py --epochs 3 --batch_size 4 --use_wandb
"""

import os
import torch
import argparse
import logging
import numpy as np
import gc
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import json
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from llama_data import load_data, create_data_splits, prepare_training_data, create_formatting_func

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLaMATrainer:
    """Handles LLaMA model training for critical thinking evaluation."""
    
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        
    def setup_model_and_tokenizer(self):
        """Load model with 4-bit quantization and setup LoRA."""
        logger.info(f"Loading model: {self.args.model_name}")
        
        # Configure 4-bit quantization (NF4 + double quantization)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",  # Scaled dot-product attention
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
                "gate_proj", "up_proj", "down_proj"       # MLP
            ],
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Add LoRA adapters
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("Model and tokenizer setup complete")
    
    def get_training_arguments(self):
        """Create SFT training arguments."""
        # Set wandb logging
        report_to = "wandb" if self.args.use_wandb else "tensorboard"
        
        # Auto-generate run name if not provided
        run_name = self.args.wandb_run_name
        if run_name is None:
            if self.args.no_just:
                 run_name = (f"llama_no_just_wrubric_sft_r{self.args.lora_r}_"
                       f"lr{self.args.learning_rate}_"
                       f"bs{self.args.batch_size}x{self.args.gradient_accumulation_steps}")
            else:
                run_name = (f"llama_wrubric_sft_r{self.args.lora_r}_"
                       f"lr{self.args.learning_rate}_"
                       f"bs{self.args.batch_size}x{self.args.gradient_accumulation_steps}")
        
        return SFTConfig(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            lr_scheduler_type=self.args.lr_scheduler_type,
            warmup_ratio=self.args.warmup_ratio,
            logging_steps=self.args.logging_steps,
            save_total_limit=self.args.save_total_limit,
            optim="paged_adamw_8bit",  # Memory efficient optimizer
            fp16=False,
            bf16=True,  # Use bfloat16 for training
            max_grad_norm=self.args.max_grad_norm,
            logging_dir=f"{self.args.output_dir}/logs",
            report_to=report_to,
            run_name=run_name,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            gradient_checkpointing=True,  # Save memory
            max_length=self.args.max_seq_length,
            packing=False,  # Don't pack multiple examples together, 
            # completion_only_loss=True,  # REMOVED: Managed manually in preprocessing
        )
    
    def preprocess_function(self, example):
        """
        Tokenize and mask user prompts for training.
        """
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
        
        # Rubric Info
        rubric_info = f"Not Applicable (0): {example['rubric_not_applicable']}\nBelow Emerging (1): {example['rubric_below_emerging']}\nEmerging (2): {example['rubric_emerging']}\nExpanding (3): {example['rubric_expanding']}\nExemplifying (4): {example['rubric_exemplifying']}"
        
        # Create user message
        if self.args.no_just:
            user_content = f"Evaluate the essay based on the given rubric for the given subskill. Assign one of these proficiency labels: not applicable, below emerging, emerging, expanding, or exemplifying.\n\nSubskill: {example['subskill_full']}\nDefinition: {example['subskill_definition']}\nRubric:\n{rubric_info}\n\nEssay:\n{example['essay_text']}"
            assistant_content = f"Label: {label_text}"
        else:
            user_content = f"Evaluate the essay based on the given rubric for the given subskill. Provide a detailed justification explaining your assessment, then assign one of these proficiency labels: not applicable, below emerging, emerging, expanding, or exemplifying.\n\nSubskill: {example['subskill_full']}\nDefinition: {example['subskill_definition']}\nRubric:\n{rubric_info}\n\nEssay:\n{example['essay_text']}"
            assistant_content = f"Justification: {example['justification']}\nLabel: {label_text}"
        
        # Full chat
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
        
        # Apply chat template
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # User prompt only (for length calculation)
        user_messages = [{"role": "user", "content": user_content}]
        user_text = self.tokenizer.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize (full)
        tokenized_full = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.args.max_seq_length,
            padding=False,
            return_tensors=None
        )
        
        # Tokenize (user) to find split point
        tokenized_user = self.tokenizer(
            user_text,
            truncation=True,
            max_length=self.args.max_seq_length,
            padding=False,
            return_tensors=None
        )
        
        input_ids = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]
        labels = list(input_ids) # Copy
        
        # Mask user part
        user_len = len(tokenized_user["input_ids"])
        if user_len > len(input_ids):
            user_len = len(input_ids)
            
        for i in range(user_len):
            labels[i] = -100
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def train(self, train_dataset, val_dataset):
        """Train the model using SFTTrainer."""
        logger.info("Starting SFT training...")
        
        training_args = self.get_training_arguments()
        
        # Convert DataFrames to Datasets
        logger.info("Formatting datasets...")
        if not isinstance(train_dataset, Dataset):
            # Ensure essay_id is string type to avoid PyArrow type errors
            train_dataset = train_dataset.copy()
            train_dataset['essay_id'] = train_dataset['essay_id'].astype(str)
            train_dataset = Dataset.from_pandas(train_dataset)
        if not isinstance(val_dataset, Dataset):
            # Ensure essay_id is string type to avoid PyArrow type errors
            val_dataset = val_dataset.copy()
            val_dataset['essay_id'] = val_dataset['essay_id'].astype(str)
            val_dataset = Dataset.from_pandas(val_dataset)
            
        # Apply preprocessing (remove original columns to save memory)
        # Disable multiprocessing to avoid CUDA fork issues
        logger.info(f"Preprocessing {len(train_dataset)} training examples...")
        train_dataset = train_dataset.map(
            self.preprocess_function,
            remove_columns=train_dataset.column_names,
            batched=False,
            num_proc=None,  # Disable multiprocessing entirely
            load_from_cache_file=False,
            desc="Tokenizing training data"
        )
        logger.info(f"Preprocessing {len(val_dataset)} validation examples...")
        val_dataset = val_dataset.map(
            self.preprocess_function,
            remove_columns=val_dataset.column_names,
            batched=False,
            num_proc=None,  # Disable multiprocessing entirely
            load_from_cache_file=False,
            desc="Tokenizing validation data"
        )
        
        # Data collator for padding
        collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            pad_to_multiple_of=8
        )
        
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            data_collator=collator,
        )
        
        # Train
        logger.info("="*80)
        logger.info("Starting training...")
        logger.info(f"  Train samples: {len(train_dataset)}")
        logger.info(f"  Val samples: {len(val_dataset)}")
        logger.info(f"  Epochs: {self.args.epochs}")
        logger.info(f"  Batch size: {self.args.batch_size}")
        logger.info(f"  Gradient accumulation: {self.args.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {self.args.batch_size * self.args.gradient_accumulation_steps}")
        logger.info("="*80)
        
        trainer.train()
        
        # Save final model
        trainer.save_model(f"{self.args.output_dir}/final_model")
        self.tokenizer.save_pretrained(f"{self.args.output_dir}/final_model")
        
        logger.info(f"Training complete. Model saved to {self.args.output_dir}/final_model")


def run_inference(args, test_df, rubric_df):
    """Run inference on the test set."""
    logger.info("Running inference on test set...")
    
    # Load model and tokenizer
    adapter_path = f"{args.output_dir}/final_model"
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    
    logger.info(f"Loading adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    # Run inference
    # Check for existing results to resume or append
    output_path = args.inference_output
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                results = json.load(f)
            logger.info(f"Loaded existing results for {len(results)} essays")
        except json.JSONDecodeError:
            results = {}
            logger.warning("Could not decode existing results file, starting fresh")
    else:
        results = {}
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Prepare subskills
    subskills = []
    for _, row in rubric_df.iterrows():
        rubric_info = f"Not Applicable (0): {row['Not Applicable']}\nBelow Emerging (1): {row['Below Emerging']}\nEmerging (2): {row['Emerging']}\nExpanding (3): {row['Expanding']}\nExemplifying (4): {row['Exemplifying']}" 
        subskills.append({
            "full": row['Subskill'],
            "definition": row['Definition'],
            "rubric_info": rubric_info
        })
    
    # Iterate through test essays
    # Unique essays only
    unique_test_essays = test_df[['essay_id', 'essay_text']].drop_duplicates(subset=['essay_id'])
    
    # Batch size
    batch_size = args.batch_size
    
    for _, essay_row in tqdm(unique_test_essays.iterrows(), total=len(unique_test_essays), desc="Inference"):
        essay_id = essay_row['essay_id']
        
        # Skip if already completed
        if essay_id in results and len(results[essay_id]) >= len(subskills):
            continue

        essay_text = essay_row['essay_text']
        results[essay_id] = results.get(essay_id, {})
        
        # Collect prompts for subskills
        prompts = []
        subskill_mapping = [] 
        
        for subskill in subskills:
            # Skip if subskill already done
            if subskill['full'] in results[essay_id]:
                continue
                
            rubric_info = subskill['rubric_info']

            # Construct prompt
            if args.no_just:
                user_content = f"Evaluate the essay for the given subskill. Assign one of these proficiency labels: not applicable, below emerging, emerging, expanding, or exemplifying.\n\nSubskill: {subskill['full']}\nDefinition: {subskill['definition']}\nRubric:\n{rubric_info}\n\nEssay:\n{essay_text}"
            else:
                user_content = f"Evaluate the essay for the given subskill. Provide a detailed justification explaining your assessment, then assign one of these proficiency labels: not applicable, below emerging, emerging, expanding, or exemplifying.\n\nSubskill: {subskill['full']}\nDefinition: {subskill['definition']}\nRubric:\n{rubric_info}\n\nEssay:\n{essay_text}"
            
            messages = [
                {"role": "user", "content": user_content}
            ]
            
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(input_text)
            subskill_mapping.append(subskill['full'])
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_subskills = subskill_mapping[i:i + batch_size]
            
            if not batch_prompts:
                continue
                
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # generated_ids = outputs[:, inputs.input_ids.shape[1]:]  # Slice input
            # Careful with slicing if padding usage varies, but with left padding inputs.input_ids.shape[1] is consistent
            generated_ids = outputs[:, inputs.input_ids.shape[1]:]
            
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Parse response
            for generated_text, subskill_name in zip(generated_texts, batch_subskills):
                justification = ""
                label = ""
                
                lower_gen = generated_text.lower()
                
                if args.no_just:
                    # In no_just mode, we expect "Label: ..."
                    if "label:" in lower_gen:
                        parts = generated_text.split("Label:")
                        label_part = parts[-1].strip().lower().rstrip(".")
                        label = label_part.split('\n')[0].strip()
                    else:
                        # Fallback if model just outputs label directly or fails format
                        # Attempt to matching known labels
                        known_labels = ["not applicable", "below emerging", "emerging", "expanding", "exemplifying"]
                        found_labels = [l for l in known_labels if l in lower_gen]
                        if found_labels:
                             # take the longest match or last one? usually just one.
                             label = found_labels[0]
                        else:
                             label = "unknown"
                    justification = "N/A (no_just mode)"
                else:
                    if "label:" in lower_gen:
                        parts = generated_text.split("Label:")
                        label_part = parts[-1].strip().lower().rstrip(".")
                        
                        label = label_part.split('\n')[0].strip()
                        
                        if "Justification:" in parts[0]:
                            try:
                                justification = parts[0].split("Justification:", 1)[1].strip()
                            except IndexError:
                                justification = parts[0].strip()
                        else:
                            justification = parts[0].strip()
                    else:
                        label = "unknown"
                        justification = generated_text
                
                results[essay_id][subskill_name] = {
                    "label": label,
                    "justification": justification
                }
            
        # Save results dynamic dump
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fine-tune Llama 3.1 8B for critical thinking evaluation'
    )
    
    # Model
    parser.add_argument(
        '--model_name',
        type=str,
        default='meta-llama/Llama-3.1-8B-Instruct',
        help='HuggingFace model name'
    )
    
    # LoRA parameters
    parser.add_argument('--lora_r', type=int, default=32, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=64, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Per-device batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine', help='LR scheduler type')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--max_grad_norm', type=float, default=0.3, help='Max gradient norm')
    parser.add_argument('--max_seq_length', type=int, default=1536, help='Maximum sequence length')
    
    # Logging and saving
    parser.add_argument('--output_dir', type=str, default='llama_results_wrubric', help='Output directory')
    parser.add_argument('--logging_steps', type=int, default=10, help='Logging steps')
    parser.add_argument('--save_total_limit', type=int, default=1, help='Maximum checkpoints to keep')
    
    # Wandb
    parser.add_argument('--wandb_project', type=str, default='critical-thinking', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb for logging')
    
    # Debug mode
    parser.add_argument('--debug', action='store_true', help='Debug mode: use fewer samples')
    parser.add_argument('--debug_samples', type=int, default=100, help='Samples in debug mode')
    
    # Inference
    parser.add_argument('--inference_only', action='store_true', help='Run inference only (no training)')
    parser.add_argument('--inference_output', type=str, default='llama_results/test_results.json', help='Output file for inference results')
    
    # New mode: No Justification
    parser.add_argument('--no_just', action='store_true', help='Train without justification in output')
    
    # Multiple Splits
    parser.add_argument('--multiple_splits', action='store_true', help='Train on multiple random splits (47, 123, 456, 789, 999)')
    
    # ModernBERT parity experiments
    parser.add_argument('--generalize', action='store_true', help='Run generalization experiment: train on 2 subskills, test on the third')
    parser.add_argument('--gen_subskills', action='store_true', help='Run subskill-based fine-tuning: split by subskills, not essays')
    parser.add_argument('--gen_train', action='store_true', help='Run inference on train+val essays (80%%) using already-trained --generalize checkpoints, writing test_results_train.json')
    parser.add_argument('--generalize_same_subskills', action='store_true', help='Run inference on the 20%% held-out test essays for the seen (train) subskills using already-trained --generalize checkpoints, writing test_results_seen.json')
    parser.add_argument('--generalize_seen', action='store_true', help='Train on seen subskills (GT 80%% + pseudo-labels 20%%), infer on unseen subskill for both test and train+val essays')

    return parser.parse_args()


def perform_training_run(args, train_data, val_data, test_df, rubric_df, output_dir=None):
    """
    Core function to train and run inference.
    
    Args:
        args: Argument parser args
        train_data: dataframe/dataset for training
        val_data: dataframe/dataset for validation
        test_df: raw dataframe of test essays (for run_inference)
        rubric_df: rubric dataframe
        output_dir: (Optional) override output directory
    """
    original_output_dir = args.output_dir
    if output_dir:
        args.output_dir = output_dir
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If inference only, just load model and run inference
    if args.inference_only:
        # Update output file for inference
        original_inf_output = args.inference_output
        args.inference_output = os.path.join(args.output_dir, "test_results.json")
        
        run_inference(args, test_df, rubric_df)
        
        args.inference_output = original_inf_output # Restore
        if output_dir: args.output_dir = original_output_dir
        return

    if args.debug:
        logger.info(f"Debug mode: limiting to {args.debug_samples} training samples")
        train_data = train_data.iloc[:args.debug_samples]
        val_data = val_data.iloc[:min(20, len(val_data))]
    
    # Train
    logger.info(f"Starting training in {args.output_dir}...")
    trainer = LLaMATrainer(args)
    trainer.setup_model_and_tokenizer()
    trainer.train(train_data, val_data)
    
    # Run inference immediately after training
    original_inf_output = args.inference_output
    args.inference_output = os.path.join(args.output_dir, "test_results.json")
    
    run_inference(args, test_df, rubric_df)
    
    args.inference_output = original_inf_output # Restore
    
    # Cleanup
    del trainer.model
    del trainer.tokenizer
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    # Restore args
    if output_dir:
        args.output_dir = original_output_dir


def train_pipeline(args, gt_df, rubric_df, justification_df, random_state):
    """Run training and inference for a specific split."""
    logger.info(f"Running pipeline with random_state={random_state}")
    
    # Create splits
    train_df, val_df, test_df = create_data_splits(gt_df, random_state=random_state)

    # log the size of the train, val, and test_df
    logger.info(f"Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")
    
    # Initialize wandb for this split if requested
    if args.use_wandb:
        import wandb
        # Generate run name if not provided or if running multiple splits
        run_name = args.wandb_run_name
        if run_name is None or args.multiple_splits:
            prefix = "llama_no_just_wrubric" if args.no_just else "llama_wrubric"
            run_name = f"{prefix}_r{args.lora_r}_rs{random_state}"
            if args.debug:
                run_name = "DEBUG_" + run_name
        
        # Important: set in args so Trainer uses it
        args.wandb_run_name = run_name

        # Ensure proper reinit
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            reinit=True
        )

    # Prepare data
    train_data = prepare_training_data(train_df, rubric_df, justification_df, 'train')
    val_data = prepare_training_data(val_df, rubric_df, justification_df, 'val')
    
    perform_training_run(args, train_data, val_data, test_df, rubric_df)
    
    if args.use_wandb:
        wandb.finish()


def filter_dataset_by_subskills(df, subskill_nums):
    """Filter dataset to include only specified subskills."""
    # Subskill column usually like "2.1: Synthesizing Multiple Sources"
    # We want to match if it starts with "2." or "3." etc. or "Subskill 2" if that format used
    
    # From modernbert: tuple([f"{num}." for num in target_subskills])
    prefixes = tuple(f"{num}." for num in subskill_nums)
    
    # Filter
    filtered_df = df[df['subskill_full'].str.startswith(prefixes)].copy()
    logger.info(f"Filtered dataset from {len(df)} to {len(filtered_df)} rows for subskills {subskill_nums}")
    return filtered_df


def prepare_pseudo_label_training_data(test_df, rubric_df, pseudo_labels, train_subskill_nums):
    """
    Build a training DataFrame from pseudo-labeled test essays (test_results_seen.json).

    Args:
        test_df: DataFrame with test essays (must have essay_id, essay_text columns)
        rubric_df: rubric DataFrame
        pseudo_labels: dict loaded from test_results_seen.json
                       {essay_id: {subskill_full: {label: ..., justification: ...}}}
        train_subskill_nums: list of seen subskill numbers (e.g., [3, 4])

    Returns:
        DataFrame with same columns as prepare_training_data output
    """
    import pandas as pd

    # Build rubric map for seen subskills only
    train_sub_prefixes = tuple(f"{num}." for num in train_subskill_nums)
    rubric_map = {}
    for _, rubric_row in rubric_df.iterrows():
        subskill_full = rubric_row['Subskill']
        if not subskill_full.startswith(train_sub_prefixes):
            continue
        rubric_map[subskill_full] = {
            'definition': rubric_row['Definition'],
            'not_applicable': rubric_row['Not Applicable'],
            'below_emerging': rubric_row['Below Emerging'],
            'emerging': rubric_row['Emerging'],
            'expanding': rubric_row['Expanding'],
            'exemplifying': rubric_row['Exemplifying'],
        }

    combined_data = []
    unique_test = test_df[['essay_id', 'essay_text']].drop_duplicates(subset=['essay_id'])

    for _, essay_row in unique_test.iterrows():
        essay_id = str(essay_row['essay_id'])
        essay_text = essay_row['essay_text']

        if essay_id not in pseudo_labels:
            continue

        for subskill_full, rubric_info in rubric_map.items():
            if subskill_full not in pseudo_labels[essay_id]:
                continue

            pred = pseudo_labels[essay_id][subskill_full]
            label = pred.get('label', '')
            justification = pred.get('justification', '')

            if not label or label == 'unknown':
                continue

            combined_data.append({
                'essay_id': essay_id,
                'essay_text': essay_text,
                'subskill_full': subskill_full,
                'subskill_definition': rubric_info['definition'],
                'rubric_not_applicable': rubric_info['not_applicable'],
                'rubric_below_emerging': rubric_info['below_emerging'],
                'rubric_emerging': rubric_info['emerging'],
                'rubric_expanding': rubric_info['expanding'],
                'rubric_exemplifying': rubric_info['exemplifying'],
                'label': label,
                'justification': justification,
            })

    result_df = pd.DataFrame(combined_data)
    logger.info(f"Built {len(result_df)} pseudo-label training rows from {len(unique_test)} test essays")
    return result_df


def run_gen_train_inference(args, gt_df, rubric_df):
    """
    For each generalize experiment, load the already-trained checkpoint and run
    inference on the 80% train+val essays (held-out subskill only).
    Results are written to test_results_train.json alongside test_results.json.
    """
    import pandas as pd

    logger.info("="*80)
    logger.info("RUNNING GEN_TRAIN INFERENCE (train+val essays, held-out subskill)")
    logger.info("="*80)

    experiments = [
        {'train_subskills': [2, 3], 'test_subskills': [4], 'name': 'subskill_4'},
        {'train_subskills': [2, 4], 'test_subskills': [3], 'name': 'subskill_3'},
        {'train_subskills': [3, 4], 'test_subskills': [2], 'name': 'subskill_2'}
    ]

    base_split_dir = args.output_dir + "_generalize" if args.output_dir in ["llama_results", "llama_results_no_just", "llama_results_wrubric", "llama_results_no_just_wrubric"] else args.output_dir

    # Same split as --generalize (random_state=47)
    train_df, val_df, _ = create_data_splits(gt_df, random_state=47)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    logger.info(f"Train+val essays: {len(train_val_df)}")

    for exp in experiments:
        exp_name = exp['name']
        exp_output_dir = os.path.join(base_split_dir, exp_name)

        if not os.path.isdir(os.path.join(exp_output_dir, "final_model")):
            logger.warning(f"No trained checkpoint found at {exp_output_dir}/final_model — skipping {exp_name}")
            continue

        logger.info(f"\nRunning gen_train inference for {exp_name}")
        logger.info(f"  Checkpoint: {exp_output_dir}/final_model")
        logger.info(f"  Test subskills: {exp['test_subskills']}")

        # Build rubric for the held-out subskill only
        test_sub_prefixes = tuple(f"{num}." for num in exp['test_subskills'])
        curr_rubric_df = rubric_df[rubric_df['Subskill'].str.startswith(test_sub_prefixes)].copy()

        # Override output_dir so run_inference loads from the right checkpoint
        original_output_dir = args.output_dir
        original_inf_output = args.inference_output

        args.output_dir = exp_output_dir
        args.inference_output = os.path.join(exp_output_dir, "test_results_train.json")

        run_inference(args, train_val_df, curr_rubric_df)

        args.output_dir = original_output_dir
        args.inference_output = original_inf_output


def run_generalization_experiments(args, gt_df, rubric_df, justification_df):
    """
    Run generalization experiment: train on 2 subskills, test on the third.
    Structure similar to modernbert_classify.py
    """
    logger.info("="*80)
    logger.info("RUNNING GENERALIZATION EXPERIMENTS")
    logger.info("Training on 2 subskills, testing on the third")
    logger.info("="*80)
    
    # Define experiments
    experiments = [
        {'train_subskills': [2, 3], 'test_subskills': [4], 'name': 'subskill_4'},
        {'train_subskills': [2, 4], 'test_subskills': [3], 'name': 'subskill_3'},
        {'train_subskills': [3, 4], 'test_subskills': [2], 'name': 'subskill_2'}
    ]
    
    base_split_dir = args.output_dir + "_generalize" if args.output_dir in ["llama_results", "llama_results_no_just", "llama_results_wrubric", "llama_results_no_just_wrubric"] else args.output_dir
    
    # Create 70-10-20 split with random_state=47 (Fixed for these experiments)
    train_df, val_df, test_df = create_data_splits(gt_df, random_state=47)
    
    # Prepare FULL datasets first
    logger.info("Preparing full datasets...")
    full_train_data = prepare_training_data(train_df, rubric_df, justification_df, 'train')
    full_val_data = prepare_training_data(val_df, rubric_df, justification_df, 'val')
    
    # For inference, normally run_inference takes test_df (essays) and rubric_df and builds prompts for ALL subskills in rubric.
    # But here we only want to test on specific subskills.
    # The run_inference function needs to be told which subskills to run.
    # NOTE: current run_inference runs on ALL subskills in rubric_df.
    # We can handle this by passing a filtered rubric_df to perform_training_run!
    
    for exp in experiments:
        exp_name = exp['name']
        logger.info(f"\nRunning Experiment: {exp_name}")
        logger.info(f"  Train: {exp['train_subskills']} -> Test: {exp['test_subskills']}")
        
        # Setup WandB
        if args.use_wandb:
            import wandb
            prefix = "llama_no_just_wrubric" if args.no_just else "llama_wrubric"
            run_name = f"{prefix}_gen_{exp_name}"
            
            # Important: Set in args so SFTTrainer uses this name instead of auto-generated one
            args.wandb_run_name = run_name
            
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={**vars(args), **exp},
                reinit=True
            )
            
        # Filter Training Data
        curr_train_data = filter_dataset_by_subskills(full_train_data, exp['train_subskills'])
        curr_val_data = filter_dataset_by_subskills(full_val_data, exp['train_subskills'])
        
        # Filter Rubric for Testing (this controls what run_inference predicts on)
        test_sub_prefixes = tuple(f"{num}." for num in exp['test_subskills'])
        curr_rubric_df = rubric_df[rubric_df['Subskill'].str.startswith(test_sub_prefixes)].copy()
        
        exp_output_dir = os.path.join(base_split_dir, exp_name)
        
        # Save experiment info
        os.makedirs(exp_output_dir, exist_ok=True)
        with open(os.path.join(exp_output_dir, "experiment_info.json"), "w") as f:
            json.dump(exp, f, indent=2)

        perform_training_run(
            args, 
            curr_train_data, 
            curr_val_data, 
            test_df, # Use full test essays, but restricted rubric
            curr_rubric_df, 
            output_dir=exp_output_dir
        )
        
        if args.use_wandb:
            wandb.finish()


def run_generalize_same_subskills_inference(args, gt_df, rubric_df):
    """
    For each generalize experiment, load the already-trained checkpoint and run
    inference on the 20% held-out TEST essays for the SEEN (train) subskills.
    Results are written to test_results_seen.json alongside test_results.json.

    e.g. for subskill_2 (trained on 3 & 4, held-out test subskill = 2):
        -> run inference on test essays for subskills 3 & 4 (the seen ones)
    """
    import pandas as pd

    logger.info("="*80)
    logger.info("RUNNING GENERALIZE_SAME_SUBSKILLS INFERENCE (test essays, seen/train subskills)")
    logger.info("="*80)

    experiments = [
        {'train_subskills': [2, 3], 'test_subskills': [4], 'name': 'subskill_4'},
        {'train_subskills': [2, 4], 'test_subskills': [3], 'name': 'subskill_3'},
        {'train_subskills': [3, 4], 'test_subskills': [2], 'name': 'subskill_2'}
    ]

    base_split_dir = args.output_dir + "_generalize" if args.output_dir in ["llama_results", "llama_results_no_just", "llama_results_wrubric", "llama_results_no_just_wrubric"] else args.output_dir

    # Same split as --generalize (random_state=47)
    _, _, test_df = create_data_splits(gt_df, random_state=47)
    logger.info(f"Test essays: {len(test_df)}")

    for exp in experiments:
        exp_name = exp['name']
        exp_output_dir = os.path.join(base_split_dir, exp_name)

        if not os.path.isdir(os.path.join(exp_output_dir, "final_model")):
            logger.warning(f"No trained checkpoint found at {exp_output_dir}/final_model — skipping {exp_name}")
            continue

        logger.info(f"\nRunning generalize_same_subskills inference for {exp_name}")
        logger.info(f"  Checkpoint: {exp_output_dir}/final_model")
        logger.info(f"  Seen (train) subskills: {exp['train_subskills']}")

        # Build rubric for the seen (train) subskills only
        train_sub_prefixes = tuple(f"{num}." for num in exp['train_subskills'])
        curr_rubric_df = rubric_df[rubric_df['Subskill'].str.startswith(train_sub_prefixes)].copy()

        original_output_dir = args.output_dir
        original_inf_output = args.inference_output

        args.output_dir = exp_output_dir
        args.inference_output = os.path.join(exp_output_dir, "test_results_seen.json")

        run_inference(args, test_df, curr_rubric_df)

        args.output_dir = original_output_dir
        args.inference_output = original_inf_output


def run_generalize_seen_experiments(args, gt_df, rubric_df, justification_df):
    """
    --generalize_seen: for each leave-one-subskill-out experiment, train on:
      - Ground-truth labels for seen subskills on 80% (train+val) essays
      - Pseudo-labels for seen subskills on 20% test essays (from test_results_seen.json)
    Then infer on the unseen subskill for:
      - 20% test essays  -> test_results.json
      - 80% train+val essays -> test_results_train.json
    Output goes to llama_results_wrubric_generalize_seen/
    """
    import pandas as pd

    logger.info("="*80)
    logger.info("RUNNING GENERALIZE_SEEN EXPERIMENTS")
    logger.info("Train: seen subskills (GT 80% + pseudo-label 20%), Test: unseen subskill")
    logger.info("="*80)

    experiments = [
        {'train_subskills': [2, 3], 'test_subskills': [4], 'name': 'subskill_4'},
        {'train_subskills': [2, 4], 'test_subskills': [3], 'name': 'subskill_3'},
        {'train_subskills': [3, 4], 'test_subskills': [2], 'name': 'subskill_2'},
    ]

    # Where existing --generalize checkpoints (and test_results_seen.json) live
    base_generalize_dir = args.output_dir + "_generalize" if args.output_dir in ["llama_results", "llama_results_no_just", "llama_results_wrubric", "llama_results_no_just_wrubric"] else args.output_dir + "_generalize"

    # Output dir for this new experiment
    base_seen_dir = args.output_dir + "_generalize_seen" if args.output_dir in ["llama_results", "llama_results_no_just", "llama_results_wrubric", "llama_results_no_just_wrubric"] else args.output_dir + "_generalize_seen"

    # Same 70/10/20 split as --generalize
    train_df, val_df, test_df = create_data_splits(gt_df, random_state=47)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Prepare full ground-truth data once
    logger.info("Preparing ground-truth training/val data...")
    full_train_data = prepare_training_data(train_df, rubric_df, justification_df, 'train')
    full_val_data = prepare_training_data(val_df, rubric_df, justification_df, 'val')

    for exp in experiments:
        exp_name = exp['name']
        logger.info(f"\nRunning Experiment: {exp_name}")
        logger.info(f"  Train subskills: {exp['train_subskills']} | Test subskill: {exp['test_subskills']}")

        # 1. GT data for seen subskills (70% train essays)
        gt_train_data = filter_dataset_by_subskills(full_train_data, exp['train_subskills'])
        gt_val_data = filter_dataset_by_subskills(full_val_data, exp['train_subskills'])

        # 2. Load pseudo-labels from test_results_seen.json
        seen_json_path = os.path.join(base_generalize_dir, exp_name, "test_results_seen.json")
        if not os.path.exists(seen_json_path):
            logger.warning(f"test_results_seen.json not found at {seen_json_path} — skipping {exp_name}")
            logger.warning("Run --generalize_same_subskills first to generate it.")
            continue

        with open(seen_json_path, 'r') as f:
            pseudo_labels = json.load(f)
        logger.info(f"  Loaded pseudo-labels for {len(pseudo_labels)} essays from {seen_json_path}")

        # 3. Build pseudo-labeled training data from 20% test essays
        pseudo_train_data = prepare_pseudo_label_training_data(
            test_df, rubric_df, pseudo_labels, exp['train_subskills']
        )

        # 4. Combine GT train + pseudo-labeled test into one training set
        combined_train_data = pd.concat([gt_train_data, pseudo_train_data], ignore_index=True)
        logger.info(f"  GT train rows: {len(gt_train_data)} | Pseudo rows: {len(pseudo_train_data)} | Combined: {len(combined_train_data)}")

        # 5. Unseen rubric for inference
        test_sub_prefixes = tuple(f"{num}." for num in exp['test_subskills'])
        unseen_rubric_df = rubric_df[rubric_df['Subskill'].str.startswith(test_sub_prefixes)].copy()

        exp_output_dir = os.path.join(base_seen_dir, exp_name)
        os.makedirs(exp_output_dir, exist_ok=True)

        with open(os.path.join(exp_output_dir, "experiment_info.json"), "w") as f:
            json.dump(exp, f, indent=2)

        if args.use_wandb:
            import wandb
            prefix = "llama_no_just_wrubric" if args.no_just else "llama_wrubric"
            args.wandb_run_name = f"{prefix}_gen_seen_{exp_name}"
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={**vars(args), **exp},
                reinit=True,
            )

        # 6. Train and run inference on 20% test essays for unseen subskill -> test_results.json
        perform_training_run(
            args,
            combined_train_data,
            gt_val_data,
            test_df,
            unseen_rubric_df,
            output_dir=exp_output_dir,
        )

        # 7. Second inference pass: 80% train+val essays for unseen subskill -> test_results_train.json
        original_output_dir = args.output_dir
        original_inf_output = args.inference_output

        args.output_dir = exp_output_dir
        args.inference_output = os.path.join(exp_output_dir, "test_results_train.json")

        run_inference(args, train_val_df, unseen_rubric_df)

        args.output_dir = original_output_dir
        args.inference_output = original_inf_output

        if args.use_wandb:
            wandb.finish()


def run_subskill_finetuning_experiments(args, gt_df, rubric_df, justification_df):
    """
    Run subskill-based fine-tuning experiments: split by subskills, not essays.
    
    Structure:
    - Training subskills: 90% train, 10% val (across ALL essays)
    - Test subskill: 100% test (across ALL essays)
    """
    logger.info("="*80)
    logger.info("RUNNING SUBSKILL FINE-TUNING EXPERIMENTS")
    logger.info("Split by subskills, not essays (Train 90%/Val 10% on train-subskills, Test 100% on test-subskill)")
    logger.info("="*80)
    
    experiments = [
        {"train_subskills": [2, 3], "test_subskill": 4, "name": "subskill_4"},
        {"train_subskills": [2, 4], "test_subskill": 3, "name": "subskill_3"},
        {"train_subskills": [3, 4], "test_subskill": 2, "name": "subskill_2"}
    ]
    
    base_split_dir = args.output_dir + "_gen_subskills" if args.output_dir in ["llama_results", "llama_results_no_just", "llama_results_wrubric", "llama_results_no_just_wrubric"] else args.output_dir
    
    # 1. Prepare ALL data (acting as if all essays are available)
    # We pass gt_df as 'train' just to get the full formatted dataset
    logger.info("Preparing full dataset from all essays...")
    all_data_df = prepare_training_data(gt_df, rubric_df, justification_df, 'all_data')
    
    for exp in experiments:
        exp_name = exp['name']
        logger.info(f"\nRunning Experiment: {exp_name}")
        
        # Setup WandB
        if args.use_wandb:
            import wandb
            prefix = "llama_no_just_wrubric" if args.no_just else "llama_wrubric"
            run_name = f"{prefix}_sub_{exp_name}"
            
            # Important: Set in args so SFTTrainer uses this name instead of auto-generated one
            args.wandb_run_name = run_name
            
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={**vars(args), **exp},
                reinit=True
            )
            
        # 1. Create Train/Val Set from Train Subskills
        train_subskills = exp['train_subskills']
        
        # Filter for train subskills
        train_sub_df = filter_dataset_by_subskills(all_data_df, train_subskills)
        
        # Shuffle with fixed seed for reproducibility
        train_sub_df = train_sub_df.sample(frac=1, random_state=47).reset_index(drop=True)
        
        # Split 90/10
        split_idx = int(0.9 * len(train_sub_df))
        train_data = train_sub_df.iloc[:split_idx]
        val_data = train_sub_df.iloc[split_idx:]
        
        logger.info(f"Train Subskills {train_subskills}: {len(train_data)} train, {len(val_data)} val samples")
        
        # 2. Setup Test Set (Test Subskill)
        test_subskill = exp['test_subskill']
        # For inference, perform_training_run expects a test_df (essays) and rubric.
        # It runs inference on all essays in test_df for subskills in rubric.
        # So we should pass ALL essays as test_df, and a rubric containing ONLY the test_subskill.
        
        test_rub_prefix = f"{test_subskill}."
        test_rubric_df = rubric_df[rubric_df['Subskill'].str.startswith(test_rub_prefix)].copy()
        
        exp_output_dir = os.path.join(base_split_dir, exp_name)
        os.makedirs(exp_output_dir, exist_ok=True)
        
        with open(os.path.join(exp_output_dir, "experiment_info.json"), "w") as f:
            json.dump(exp, f, indent=2)
            
        perform_training_run(
            args,
            train_data,
            val_data,
            gt_df, # Use ALL essays for testing
            test_rubric_df,
            output_dir=exp_output_dir
        )
        
        if args.use_wandb:
            wandb.finish()

def main():
    """Main function."""
    args = parse_arguments()
    
    # Adjust default output dir for no_just
    if args.no_just and args.output_dir == 'llama_results_wrubric':
        args.output_dir = 'llama_results_no_just_wrubric'
        
    base_output_dir = args.output_dir
    
    logger.info("="*80)
    logger.info("LLaMA Critical Thinking Training Configuration:")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Base Output Directory: {base_output_dir}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Mode: {'No Justification' if args.no_just else 'With Justification'}")
    logger.info(f"  Multiple Splits: {args.multiple_splits}")
    logger.info("="*80)
    
    # Load data once
    logger.info("Loading data...")
    gt_df, rubric_df, justification_df = load_data()
    
    if args.multiple_splits:
        random_states = [47, 123, 456, 789, 999]
        base_split_dir = base_output_dir + "_splits"
        logger.info(f"Running multiple splits: {random_states}")
        logger.info(f"Base output directory: {base_split_dir}")
        
        for rs in random_states:
            # Set specific output dir
            args.output_dir = os.path.join(base_split_dir, f"rstate_{rs}")
            os.makedirs(args.output_dir, exist_ok=True)
            
            try:
                train_pipeline(args, gt_df, rubric_df, justification_df, rs)
            except Exception as e:
                logger.error(f"Error in split {rs}: {e}", exc_info=True)
                continue
    
    elif args.generalize:
        # Generalization experiment
        run_generalization_experiments(args, gt_df, rubric_df, justification_df)

    elif args.gen_subskills:
        # Subskill fine-tuning experiment
        run_subskill_finetuning_experiments(args, gt_df, rubric_df, justification_df)

    elif args.gen_train:
        # Inference on train+val essays using existing --generalize checkpoints
        run_gen_train_inference(args, gt_df, rubric_df)

    elif args.generalize_same_subskills:
        # Inference on held-out test essays for seen (train) subskills
        run_generalize_same_subskills_inference(args, gt_df, rubric_df)

    elif args.generalize_seen:
        # Train on GT 80% + pseudo-labeled 20%, infer on unseen subskill
        run_generalize_seen_experiments(args, gt_df, rubric_df, justification_df)

    else:
        # Single run (default behavior)
        os.makedirs(args.output_dir, exist_ok=True)
        train_pipeline(args, gt_df, rubric_df, justification_df, 47) # Default random state 47

if __name__ == "__main__":
    main()
