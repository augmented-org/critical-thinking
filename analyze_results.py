'''
Analyze predicted results
'''

import pandas as pd 
import json
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import krippendorff
from llama_data import create_data_splits

def process_single_model_results(pred_results, gt_df, results_dir):
    """
    Process results for a single model and save to directory
    Returns: (overall_accuracy, overall_rmse, subskill_metrics)
    """
    # Create dataframe for analysis
    results_data = []
    
    # Use the same subskill mapping as the main analysis function
    subskill_map = {
        '2.1': '2.1: Synthesizing Multiple Sources',
        '2.2': '2.2: Evaluating Evidence Strength',
        '3.1': '3.1: Using Counterarguments',
        '3.2': '3.2: Using Facts and Opinions',
        '4.1': '4.1: Drawing Conclusions',
        '4.2': '4.2: Using Logical Fallacies'
    }
    
    # Use the same logic as the main analysis function
    for _, row in gt_df.iterrows():
        essay_id = row['essay_id']
        if essay_id not in pred_results:
            continue
        pred_rubric = pred_results[essay_id]
        for subskill, gt_label in row.items():
            if 'Subskill' not in subskill:
                continue
            subskill_number = subskill.split('Subskill ')[-1]
            subskill_name = subskill_map.get(subskill_number, subskill)
            pred_label = pred_rubric.get(subskill_name, {}).get('label', None)
            if pred_label not in ['Few-shot Example', 'Parsing Error'] and pred_label is not None:
                results_data.append({
                    'Essay ID': essay_id,
                    'Subskill': subskill,
                    'Ground Truth': gt_label.lower().strip(),
                    'Predicted': pred_label.lower().strip()
                })
    
    results_df = pd.DataFrame(results_data)
    
    # Process subskill-wise results
    subskill_metrics = process_subskill_metrics(results_df, "", "", results_dir)
    
    # Calculate overall metrics
    overall_accuracy = (results_df['Ground Truth'] == results_df['Predicted']).mean()
    overall_rmse = calculate_rmse(results_df['Ground Truth'], results_df['Predicted'])
    overall_alpha = calculate_krippendorff_alpha(results_df['Ground Truth'], results_df['Predicted'])
    
    # Generate overall classification report and confusion matrix
    overall_report = classification_report(results_df['Ground Truth'], results_df['Predicted'], zero_division=0)
    overall_cm = confusion_matrix(results_df['Ground Truth'], results_df['Predicted'])
    
    # Save overall results
    save_overall_results(results_dir, overall_accuracy, overall_rmse, overall_alpha, len(results_df), 
                        overall_report, overall_cm, subskill_metrics)
    
    return overall_accuracy, overall_rmse, overall_alpha, subskill_metrics

def save_overall_results(results_dir, overall_accuracy, overall_rmse, overall_alpha, total_samples, 
                        overall_report, overall_cm, subskill_metrics):
    """
    Save overall results to file
    """
    overall_file = f"{results_dir}/subskill_wise_results/overall.txt"
    os.makedirs(os.path.dirname(overall_file), exist_ok=True)
    
    with open(overall_file, 'w', encoding='utf-8') as f:
        f.write("OVERALL RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n")
        f.write(f"Overall RMSE: {overall_rmse:.4f}\n")
        f.write(f"Overall Krippendorff's α: {overall_alpha:.4f}\n")
        f.write(f"Total samples: {total_samples}\n\n")
        
        f.write("CLASSIFICATION REPORT:\n")
        f.write("-" * 40 + "\n")
        f.write(overall_report + "\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write("-" * 40 + "\n")
        f.write(str(overall_cm) + "\n\n")
        
        f.write("BREAKDOWN BY SUBSKILL:\n")
        f.write("-" * 40 + "\n\n")
        
        # Create table header
        f.write("| Subskill | Accuracy | RMSE | Krippendorff's α | F1-macro | F1-weighted | Total Samples |\n")
        f.write("|----------|----------|------|------------------|----------|-------------|---------------|\n")
        
        # Define the desired order of subskills
        subskill_order = ['Subskill 2.1', 'Subskill 2.2', 'Subskill 3.1', 'Subskill 3.2', 'Subskill 4.1', 'Subskill 4.2']
        
        # Write table rows in the specified order
        for subskill in subskill_order:
            if subskill in subskill_metrics:
                metrics = subskill_metrics[subskill]
                subskill_name = subskill.replace("Subskill ", "")  # Clean up subskill name
                accuracy = metrics.get('accuracy', 0)
                rmse = metrics.get('rmse', 0)
                alpha = metrics.get('alpha', 0)
                total_samples = metrics.get('total_samples', 0)
                f1_macro = metrics.get('f1_macro', 0)
                f1_weighted = metrics.get('f1_weighted', 0)
                
                f.write(f"| {subskill_name:<8} | {accuracy:>8.4f} | {rmse:>4.4f} | {alpha:>15.4f} | {f1_macro:>8.4f} | {f1_weighted:>11.4f} | {total_samples:>13} |\n")
        
        f.write("\n")

def analyze_split_results(model_type="bert", wrubric=False):
    """
    Analyze results from multiple random splits for BERT or ModernBERT
    """
    model_name = model_type.upper()
    if wrubric:
        model_name += "_WRUBRIC"
        
    print("=" * 80)
    print(f"ANALYZING {model_name} RESULTS FROM MULTIPLE RANDOM SPLITS")
    print("=" * 80)
    
    if model_type == "llama_no_just":
        base_dir = "llama_results_no_just_wrubric_splits" if wrubric else "llama_results_no_just_splits"
    elif model_type == "llama":
        base_dir = "llama_results_wrubric_splits" if wrubric else "llama_results_splits"
    elif model_type == "modernbert":
        base_dir = "modernbert_results_wrubric_splits" if wrubric else "modernbert_results_splits"
    else:
        base_dir = f"{model_type}_results_splits"
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} not found!")
        print(f"Please run: python {model_type}_classify.py --multiple-splits first")
        return
    
    # Load ground truth data
    gt_df = pd.read_excel("combined_essays.xlsx")
    gt_df.columns = gt_df.iloc[0]
    gt_df = gt_df[1:]
    
    # Find all split directories
    split_dirs = [d for d in os.listdir(base_dir) if d.startswith('rstate_') and os.path.isdir(os.path.join(base_dir, d))]
    split_dirs.sort()  # Sort for consistent ordering
    
    if not split_dirs:
        print(f"Error: No split directories found in {base_dir}")
        return
    
    print(f"Found {len(split_dirs)} split directories: {split_dirs}")
    
    # Process each split
    all_accuracies = []
    all_rmses = []
    all_alphas = []  # Add Krippendorff's alpha collection
    all_f1_macros = []  # Add F1 macro collection
    all_f1_weighteds = []  # Add F1 weighted collection
    all_subskill_metrics = {}
    all_predictions = []  # Store all predictions for overall F1 calculation
    all_ground_truth = []  # Store all ground truth for overall F1 calculation
    
    for split_dir in split_dirs:
        split_path = os.path.join(base_dir, split_dir)
        results_file = os.path.join(split_path, "test_results.json")
        
        print(f"\nProcessing {split_dir}...")
        
        if not os.path.exists(results_file):
            print(f"Warning: {results_file} not found, skipping...")
            continue
        
        # Load results for this split
        with open(results_file, "r", encoding="utf-8") as f:
            pred_results = json.load(f)
        
        # Process this split's results and get the results dataframe
        accuracy, rmse, alpha, subskill_metrics = process_single_model_results(pred_results, gt_df, split_path)
        
        # Calculate F1 scores for this split
        from sklearn.metrics import classification_report
        split_results_data = []
        subskill_map = {
            '2.1': '2.1: Synthesizing Multiple Sources',
            '2.2': '2.2: Evaluating Evidence Strength',
            '3.1': '3.1: Using Counterarguments',
            '3.2': '3.2: Using Facts and Opinions',
            '4.1': '4.1: Drawing Conclusions',
            '4.2': '4.2: Using Logical Fallacies'
        }
        
        # Collect this split's data for F1 calculation
        split_ground_truth = []
        split_predictions = []
        
        for _, row in gt_df.iterrows():
            essay_id = row['essay_id']
            if essay_id not in pred_results:
                continue
            pred_rubric = pred_results[essay_id]
            for subskill, gt_label in row.items():
                if 'Subskill' not in subskill:
                    continue
                subskill_number = subskill.split('Subskill ')[-1]
                subskill_name = subskill_map.get(subskill_number, subskill)
                pred_label = pred_rubric.get(subskill_name, {}).get('label', None)
                if pred_label not in ['Few-shot Example', 'Parsing Error'] and pred_label is not None:
                    split_ground_truth.append(gt_label.lower().strip())
                    split_predictions.append(pred_label.lower().strip())
                    all_ground_truth.append(gt_label.lower().strip())
                    all_predictions.append(pred_label.lower().strip())
        
        # Calculate F1 scores for this split
        if split_ground_truth and split_predictions:
            split_report = classification_report(split_ground_truth, split_predictions, 
                                               zero_division=0, output_dict=True)
            split_f1_macro = split_report.get('macro avg', {}).get('f1-score', 0)
            split_f1_weighted = split_report.get('weighted avg', {}).get('f1-score', 0)
        else:
            split_f1_macro = 0
            split_f1_weighted = 0
        
        all_accuracies.append(accuracy)
        all_rmses.append(rmse)
        all_alphas.append(alpha)
        all_f1_macros.append(split_f1_macro)
        all_f1_weighteds.append(split_f1_weighted)
        
        # Collect subskill metrics
        for subskill, metrics in subskill_metrics.items():
            if subskill not in all_subskill_metrics:
                all_subskill_metrics[subskill] = {
                    'accuracies': [],
                    'rmses': [],
                    'alphas': [],
                    'f1_macros': [],
                    'f1_weighteds': [],
                    'precision_macros': [],
                    'precision_weighteds': [],
                    'recall_macros': [],
                    'recall_weighteds': []
                }
            
            all_subskill_metrics[subskill]['accuracies'].append(metrics.get('accuracy', 0))
            all_subskill_metrics[subskill]['rmses'].append(metrics.get('rmse', 0))
            all_subskill_metrics[subskill]['alphas'].append(metrics.get('alpha', 0))
            all_subskill_metrics[subskill]['f1_macros'].append(metrics.get('f1_macro', 0))
            all_subskill_metrics[subskill]['f1_weighteds'].append(metrics.get('f1_weighted', 0))
            all_subskill_metrics[subskill]['precision_macros'].append(metrics.get('precision_macro', 0))
            all_subskill_metrics[subskill]['precision_weighteds'].append(metrics.get('precision_weighted', 0))
            all_subskill_metrics[subskill]['recall_macros'].append(metrics.get('recall_macro', 0))
            all_subskill_metrics[subskill]['recall_weighteds'].append(metrics.get('recall_weighted', 0))
        
        print(f"  Accuracy: {accuracy:.4f}, RMSE: {rmse:.4f}, Krippendorff's α: {alpha:.4f}, F1-macro: {split_f1_macro:.4f}, F1-weighted: {split_f1_weighted:.4f}")
    
    # Create overall results directory
    overall_results_dir = os.path.join(base_dir, "overall_results")
    os.makedirs(overall_results_dir, exist_ok=True)
    
    # Calculate statistics across all splits
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    mean_rmse = np.mean(all_rmses)
    std_rmse = np.std(all_rmses)
    mean_alpha = np.mean(all_alphas)
    std_alpha = np.std(all_alphas)
    mean_f1_macro = np.mean(all_f1_macros)
    std_f1_macro = np.std(all_f1_macros)
    mean_f1_weighted = np.mean(all_f1_weighteds)
    std_f1_weighted = np.std(all_f1_weighteds)
    
    # Calculate overall F1 scores and other metrics from combined predictions
    from sklearn.metrics import classification_report
    overall_report_dict = classification_report(all_ground_truth, all_predictions, 
                                               zero_division=0, output_dict=True)
    overall_macro = overall_report_dict.get('macro avg', {})
    overall_weighted = overall_report_dict.get('weighted avg', {})
    
    overall_precision_macro = overall_macro.get('precision', 0)
    overall_recall_macro = overall_macro.get('recall', 0)
    overall_f1_macro = overall_macro.get('f1-score', 0)
    overall_precision_weighted = overall_weighted.get('precision', 0)
    overall_recall_weighted = overall_weighted.get('recall', 0)
    overall_f1_weighted = overall_weighted.get('f1-score', 0)
    overall_rmse_combined = calculate_rmse(all_ground_truth, all_predictions)
    overall_alpha_combined = calculate_krippendorff_alpha(all_ground_truth, all_predictions)
    
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS ACROSS ALL SPLITS")
    print("="*80)
    print(f"Overall Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Overall RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    print(f"Overall Krippendorff's α: {mean_alpha:.4f} ± {std_alpha:.4f}")
    print(f"Overall F1-macro: {mean_f1_macro:.4f} ± {std_f1_macro:.4f}")
    print(f"Overall F1-weighted: {mean_f1_weighted:.4f} ± {std_f1_weighted:.4f}")
    print(f"Number of splits: {len(all_accuracies)}")
    print(f"\nCOMBINED CLASSIFICATION METRICS:")
    print(f"Precision (macro): {overall_precision_macro:.4f}")
    print(f"Recall (macro): {overall_recall_macro:.4f}")
    print(f"F1-score (macro): {overall_f1_macro:.4f}")
    print(f"Precision (weighted): {overall_precision_weighted:.4f}")
    print(f"Recall (weighted): {overall_recall_weighted:.4f}")
    print(f"F1-score (weighted): {overall_f1_weighted:.4f}")
    print(f"RMSE (combined): {overall_rmse_combined:.4f}")
    print(f"Krippendorff's α (combined): {overall_alpha_combined:.4f}")
    
    # Save overall results with standard deviations
    overall_file = os.path.join(overall_results_dir, "overall_statistics.txt")
    
    with open(overall_file, 'w', encoding='utf-8') as f:
        f.write(f"{model_name} MULTIPLE SPLITS - OVERALL STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Number of splits: {len(all_accuracies)}\n")
        f.write(f"Split directories: {', '.join(split_dirs)}\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Overall Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}\n")
        f.write(f"Overall RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}\n")
        f.write(f"Overall Krippendorff's α: {mean_alpha:.4f} ± {std_alpha:.4f}\n")
        f.write(f"Overall F1-macro: {mean_f1_macro:.4f} ± {std_f1_macro:.4f}\n")
        f.write(f"Overall F1-weighted: {mean_f1_weighted:.4f} ± {std_f1_weighted:.4f}\n\n")
        
        f.write("COMBINED CLASSIFICATION METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Precision (macro): {overall_precision_macro:.4f}\n")
        f.write(f"Recall (macro): {overall_recall_macro:.4f}\n")
        f.write(f"F1-score (macro): {overall_f1_macro:.4f}\n")
        f.write(f"Precision (weighted): {overall_precision_weighted:.4f}\n")
        f.write(f"Recall (weighted): {overall_recall_weighted:.4f}\n")
        f.write(f"F1-score (weighted): {overall_f1_weighted:.4f}\n")
        f.write(f"RMSE (combined): {overall_rmse_combined:.4f}\n\n")
        
        f.write("INDIVIDUAL SPLIT RESULTS:\n")
        f.write("-" * 40 + "\n")
        for i, (split_dir, acc, rmse, alpha, f1_m, f1_w) in enumerate(zip(split_dirs, all_accuracies, all_rmses, all_alphas, all_f1_macros, all_f1_weighteds)):
            f.write(f"{split_dir}: Accuracy={acc:.4f}, RMSE={rmse:.4f}, α={alpha:.4f}, F1-macro={f1_m:.4f}, F1-weighted={f1_w:.4f}\n")
        f.write("\n")
        
        f.write("BREAKDOWN BY SUBSKILL:\n")
        f.write("-" * 40 + "\n\n")
        
        # Create detailed subskill table with standard deviations
        f.write("| Subskill | Accuracy±Std | RMSE±Std | α±Std | F1-macro±Std | F1-weighted±Std |\n")
        f.write("|----------|--------------|----------|-------|--------------|----------------|\n")
        
        subskill_order = ['Subskill 2.1', 'Subskill 2.2', 'Subskill 3.1', 'Subskill 3.2', 'Subskill 4.1', 'Subskill 4.2']
        
        for subskill in subskill_order:
            if subskill in all_subskill_metrics:
                metrics = all_subskill_metrics[subskill]
                subskill_name = subskill.replace("Subskill ", "")
                
                acc_mean = np.mean(metrics['accuracies'])
                acc_std = np.std(metrics['accuracies'])
                rmse_mean = np.mean(metrics['rmses'])
                rmse_std = np.std(metrics['rmses'])
                alpha_mean = np.mean(metrics['alphas'])
                alpha_std = np.std(metrics['alphas'])
                f1m_mean = np.mean(metrics['f1_macros'])
                f1m_std = np.std(metrics['f1_macros'])
                f1w_mean = np.mean(metrics['f1_weighteds'])
                f1w_std = np.std(metrics['f1_weighteds'])
                
                f.write(f"| {subskill_name:<8} | {acc_mean:.3f}±{acc_std:.3f}   | {rmse_mean:.3f}±{rmse_std:.3f} | {alpha_mean:.3f}±{alpha_std:.3f}   | {f1m_mean:.3f}±{f1m_std:.3f}   | {f1w_mean:.3f}±{f1w_std:.3f}    |\n")
        
        f.write("\n")
    
    # Also save CSV with detailed statistics
    csv_data = []
    for subskill in subskill_order:
        if subskill in all_subskill_metrics:
            metrics = all_subskill_metrics[subskill]
            csv_data.append({
                'Subskill': subskill.replace("Subskill ", ""),
                'Accuracy_Mean': np.mean(metrics['accuracies']),
                'Accuracy_Std': np.std(metrics['accuracies']),
                'RMSE_Mean': np.mean(metrics['rmses']),
                'RMSE_Std': np.std(metrics['rmses']),
                'Precision_Macro_Mean': np.mean(metrics['precision_macros']),
                'Precision_Macro_Std': np.std(metrics['precision_macros']),
                'Recall_Macro_Mean': np.mean(metrics['recall_macros']),
                'Recall_Macro_Std': np.std(metrics['recall_macros']),
                'F1_Macro_Mean': np.mean(metrics['f1_macros']),
                'F1_Macro_Std': np.std(metrics['f1_macros']),
                'Precision_Weighted_Mean': np.mean(metrics['precision_weighteds']),
                'Precision_Weighted_Std': np.std(metrics['precision_weighteds']),
                'Recall_Weighted_Mean': np.mean(metrics['recall_weighteds']),
                'Recall_Weighted_Std': np.std(metrics['recall_weighteds']),
                'F1_Weighted_Mean': np.mean(metrics['f1_weighteds']),
                'F1_Weighted_Std': np.std(metrics['f1_weighteds'])
            })
    
    # Add overall combined metrics to CSV
    csv_data.append({
        'Subskill': 'OVERALL_COMBINED',
        'Accuracy_Mean': mean_accuracy,
        'Accuracy_Std': std_accuracy,
        'RMSE_Mean': overall_rmse_combined,
        'RMSE_Std': 0,  # Single value, no std
        'Precision_Macro_Mean': overall_precision_macro,
        'Precision_Macro_Std': 0,
        'Recall_Macro_Mean': overall_recall_macro,
        'Recall_Macro_Std': 0,
        'F1_Macro_Mean': overall_f1_macro,
        'F1_Macro_Std': 0,
        'Precision_Weighted_Mean': overall_precision_weighted,
        'Precision_Weighted_Std': 0,
        'Recall_Weighted_Mean': overall_recall_weighted,
        'Recall_Weighted_Std': 0,
        'F1_Weighted_Mean': overall_f1_weighted,
        'F1_Weighted_Std': 0
    })
    
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(os.path.join(overall_results_dir, "subskill_statistics.csv"), index=False)
    
    # Save simple JSON file for statistical significance testing
    simple_results = {
        "model_type": model_type,
        "accuracies": [float(acc) for acc in all_accuracies],
        "rmses": [float(rmse) for rmse in all_rmses],
        "f1_macros": [float(f1_m) for f1_m in all_f1_macros],
        "f1_weighteds": [float(f1_w) for f1_w in all_f1_weighteds]
    }
    
    json_file = os.path.join(overall_results_dir, f"{model_type}_results.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(simple_results, f, indent=2)
    
    print(f"\nOverall results saved to: {overall_results_dir}")
    print(f"  - overall_statistics.txt: Detailed report with standard deviations")
    print(f"  - subskill_statistics.csv: Machine-readable statistics")
    print(f"  - {model_type}_results.json: Simple JSON for significance testing")

def analyze_ft_split_results(model_choice=0, few_shot=False, k=1):
    """
    Analyze LLM results (zero_shot/few_shot) on the subset of data used for fine-tuning,
    for each random split separately (using bert_results_splits test sets)
    """
    # ANALYZE_FT_SPLIT_RESULTS function - alpha integration
    # Determine model prefix and result suffix
    if model_choice == 0:
        prefix = "gpt-4.1"
    elif model_choice == 1:
        prefix = "gpt-5"
    elif model_choice == 2:
        prefix = "gpt-5-mini"
    
    if few_shot:
        result_suffix = "few_shot" if k == 1 else f"few_shot_k{k}"
    else:
        result_suffix = "zero_shot"
    
    print("=" * 80)
    print(f"ANALYZING {prefix.upper()} {result_suffix.upper()} RESULTS ON FT SPLITS")
    print("=" * 80)
    
    # Check if LLM results exist
    llm_results_dir = f"{prefix}_results/results_{result_suffix}"
    llm_results_file = f"{llm_results_dir}/classification_results.json"
    
    if not os.path.exists(llm_results_file):
        print(f"Error: LLM results file {llm_results_file} not found!")
        print(f"Please run the {prefix} {result_suffix} experiment first")
        return
    
    # Load LLM results
    with open(llm_results_file, "r", encoding="utf-8") as f:
        llm_pred_results = json.load(f)
    
    # Check if BERT splits exist
    bert_splits_dir = "modernbert_results_wrubric_splits"
    if not os.path.exists(bert_splits_dir):
        print(f"Error: Directory {bert_splits_dir} not found!")
        print("Please run: python bert_classify.py --multiple-splits first")
        return
    
    # Load ground truth data
    gt_df = pd.read_excel("combined_essays.xlsx")
    gt_df.columns = gt_df.iloc[0]
    gt_df = gt_df[1:]
    
    # Find all split directories
    split_dirs = [d for d in os.listdir(bert_splits_dir) if d.startswith('rstate_') and os.path.isdir(os.path.join(bert_splits_dir, d))]
    split_dirs.sort()  # Sort for consistent ordering
    
    if not split_dirs:
        print(f"Error: No split directories found in {bert_splits_dir}")
        return
    
    print(f"Found {len(split_dirs)} split directories: {split_dirs}")
    
    # Create results directory for ft_mode splits
    ft_splits_dir = f"{prefix}_results/results_{result_suffix}_ft_splits"
    os.makedirs(ft_splits_dir, exist_ok=True)
    
    # Process each split
    all_accuracies = []
    all_rmses = []
    all_alphas = []  # Add Krippendorff's alpha collection
    all_f1_macros = []  # Add F1 macro collection
    all_f1_weighteds = []  # Add F1 weighted collection
    all_subskill_metrics = {}
    all_predictions = []  # Store all predictions for overall F1 calculation
    all_ground_truth = []  # Store all ground truth for overall F1 calculation
    
    for split_dir in split_dirs:
        split_path = os.path.join(bert_splits_dir, split_dir)
        bert_results_file = os.path.join(split_path, "test_results.json")
        
        print(f"\nProcessing {split_dir}...")
        
        if not os.path.exists(bert_results_file):
            print(f"Warning: {bert_results_file} not found, skipping...")
            continue
        
        # Load BERT results to get the test set essay IDs for this split
        with open(bert_results_file, "r", encoding="utf-8") as f:
            bert_split_results = json.load(f)
        
        # Filter LLM results to only include essays in this split's test set
        filtered_llm_results = {k: v for k, v in llm_pred_results.items() if k in bert_split_results}
        
        print(f"  LLM total essays: {len(llm_pred_results)}")
        print(f"  BERT split test essays: {len(bert_split_results)}")
        print(f"  Filtered LLM essays: {len(filtered_llm_results)}")
        
        # Create results directory for this split
        split_results_dir = os.path.join(ft_splits_dir, split_dir)
        os.makedirs(split_results_dir, exist_ok=True)
        
        # Process this split's results
        accuracy, rmse, alpha, subskill_metrics = process_single_model_results(filtered_llm_results, gt_df, split_results_dir)
        
        # Calculate F1 scores for this split
        from sklearn.metrics import classification_report
        split_ground_truth = []
        split_predictions = []
        
        # Also get the detailed results for overall F1 calculation
        subskill_map = {
            '2.1': '2.1: Synthesizing Multiple Sources',
            '2.2': '2.2: Evaluating Evidence Strength',
            '3.1': '3.1: Using Counterarguments',
            '3.2': '3.2: Using Facts and Opinions',
            '4.1': '4.1: Drawing Conclusions',
            '4.2': '4.2: Using Logical Fallacies'
        }
        
        for _, row in gt_df.iterrows():
            essay_id = row['essay_id']
            if essay_id not in filtered_llm_results:
                continue
            pred_rubric = filtered_llm_results[essay_id]
            for subskill, gt_label in row.items():
                if 'Subskill' not in subskill:
                    continue
                subskill_number = subskill.split('Subskill ')[-1]
                subskill_name = subskill_map.get(subskill_number, subskill)
                pred_label = pred_rubric.get(subskill_name, {}).get('label', None)
                if pred_label not in ['Few-shot Example', 'Parsing Error'] and pred_label is not None:
                    split_ground_truth.append(gt_label.lower().strip())
                    split_predictions.append(pred_label.lower().strip())
                    all_ground_truth.append(gt_label.lower().strip())
                    all_predictions.append(pred_label.lower().strip())
        
        # Calculate F1 scores for this split
        if split_ground_truth and split_predictions:
            split_report = classification_report(split_ground_truth, split_predictions, 
                                               zero_division=0, output_dict=True)
            split_f1_macro = split_report.get('macro avg', {}).get('f1-score', 0)
            split_f1_weighted = split_report.get('weighted avg', {}).get('f1-score', 0)
        else:
            split_f1_macro = 0
            split_f1_weighted = 0
        
        all_accuracies.append(accuracy)
        all_rmses.append(rmse)
        all_alphas.append(alpha)
        all_f1_macros.append(split_f1_macro)
        all_f1_weighteds.append(split_f1_weighted)
        
        # Collect subskill metrics
        for subskill, metrics in subskill_metrics.items():
            if subskill not in all_subskill_metrics:
                all_subskill_metrics[subskill] = {
                    'accuracies': [],
                    'rmses': [],
                    'alphas': [],
                    'f1_macros': [],
                    'f1_weighteds': [],
                    'precision_macros': [],
                    'precision_weighteds': [],
                    'recall_macros': [],
                    'recall_weighteds': []
                }
            
            all_subskill_metrics[subskill]['accuracies'].append(metrics.get('accuracy', 0))
            all_subskill_metrics[subskill]['rmses'].append(metrics.get('rmse', 0))
            all_subskill_metrics[subskill]['alphas'].append(metrics.get('alpha', 0))
            all_subskill_metrics[subskill]['f1_macros'].append(metrics.get('f1_macro', 0))
            all_subskill_metrics[subskill]['f1_weighteds'].append(metrics.get('f1_weighted', 0))
            all_subskill_metrics[subskill]['precision_macros'].append(metrics.get('precision_macro', 0))
            all_subskill_metrics[subskill]['precision_weighteds'].append(metrics.get('precision_weighted', 0))
            all_subskill_metrics[subskill]['recall_macros'].append(metrics.get('recall_macro', 0))
            all_subskill_metrics[subskill]['recall_weighteds'].append(metrics.get('recall_weighted', 0))
        
        print(f"  Accuracy: {accuracy:.4f}, RMSE: {rmse:.4f}, Krippendorff's α: {alpha:.4f}, F1-macro: {split_f1_macro:.4f}, F1-weighted: {split_f1_weighted:.4f}")
    
    # Create overall results directory
    overall_results_dir = os.path.join(ft_splits_dir, "overall_results")
    os.makedirs(overall_results_dir, exist_ok=True)
    
    # Calculate statistics across all splits
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    mean_rmse = np.mean(all_rmses)
    std_rmse = np.std(all_rmses)
    mean_alpha = np.mean(all_alphas)
    std_alpha = np.std(all_alphas)
    mean_f1_macro = np.mean(all_f1_macros)
    std_f1_macro = np.std(all_f1_macros)
    mean_f1_weighted = np.mean(all_f1_weighteds)
    std_f1_weighted = np.std(all_f1_weighteds)
    
    # Calculate overall F1 scores and other metrics from combined predictions
    from sklearn.metrics import classification_report
    overall_report_dict = classification_report(all_ground_truth, all_predictions, 
                                               zero_division=0, output_dict=True)
    overall_macro = overall_report_dict.get('macro avg', {})
    overall_weighted = overall_report_dict.get('weighted avg', {})
    
    overall_precision_macro = overall_macro.get('precision', 0)
    overall_recall_macro = overall_macro.get('recall', 0)
    overall_f1_macro = overall_macro.get('f1-score', 0)
    overall_precision_weighted = overall_weighted.get('precision', 0)
    overall_recall_weighted = overall_weighted.get('recall', 0)
    overall_f1_weighted = overall_weighted.get('f1-score', 0)
    overall_rmse_combined = calculate_rmse(all_ground_truth, all_predictions)
    
    print(f"\n{'='*80}")
    print("OVERALL FT_SPLIT STATISTICS ACROSS ALL SPLITS")
    print("="*80)
    print(f"Overall Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Overall RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    print(f"Overall Krippendorff's α: {mean_alpha:.4f} ± {std_alpha:.4f}")
    print(f"Overall F1-macro: {mean_f1_macro:.4f} ± {std_f1_macro:.4f}")
    print(f"Overall F1-weighted: {mean_f1_weighted:.4f} ± {std_f1_weighted:.4f}")
    print(f"Number of splits: {len(all_accuracies)}")
    print(f"\nCOMBINED CLASSIFICATION METRICS:")
    print(f"Precision (macro): {overall_precision_macro:.4f}")
    print(f"Recall (macro): {overall_recall_macro:.4f}")
    print(f"F1-score (macro): {overall_f1_macro:.4f}")
    print(f"Precision (weighted): {overall_precision_weighted:.4f}")
    print(f"Recall (weighted): {overall_recall_weighted:.4f}")
    print(f"F1-score (weighted): {overall_f1_weighted:.4f}")
    print(f"RMSE (combined): {overall_rmse_combined:.4f}")
    
    # Save overall results with standard deviations
    overall_file = os.path.join(overall_results_dir, "overall_statistics.txt")
    
    with open(overall_file, 'w', encoding='utf-8') as f:
        f.write(f"{prefix.upper()} {result_suffix.upper()} FT_SPLITS - OVERALL STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Model: {prefix}\n")
        f.write(f"Mode: {result_suffix}\n")
        f.write(f"Number of splits: {len(all_accuracies)}\n")
        f.write(f"Split directories: {', '.join(split_dirs)}\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Overall Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}\n")
        f.write(f"Overall RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}\n")
        f.write(f"Overall Krippendorff's α: {mean_alpha:.4f} ± {std_alpha:.4f}\n")
        f.write(f"Overall F1-macro: {mean_f1_macro:.4f} ± {std_f1_macro:.4f}\n")
        f.write(f"Overall F1-weighted: {mean_f1_weighted:.4f} ± {std_f1_weighted:.4f}\n\n")
        
        f.write("COMBINED CLASSIFICATION METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Precision (macro): {overall_precision_macro:.4f}\n")
        f.write(f"Recall (macro): {overall_recall_macro:.4f}\n")
        f.write(f"F1-score (macro): {overall_f1_macro:.4f}\n")
        f.write(f"Precision (weighted): {overall_precision_weighted:.4f}\n")
        f.write(f"Recall (weighted): {overall_recall_weighted:.4f}\n")
        f.write(f"F1-score (weighted): {overall_f1_weighted:.4f}\n")
        f.write(f"RMSE (combined): {overall_rmse_combined:.4f}\n\n")
        
        f.write("INDIVIDUAL SPLIT RESULTS:\n")
        f.write("-" * 40 + "\n")
        for i, (split_dir, acc, rmse, alpha, f1_m, f1_w) in enumerate(zip(split_dirs, all_accuracies, all_rmses, all_alphas, all_f1_macros, all_f1_weighteds)):
            f.write(f"{split_dir}: Accuracy={acc:.4f}, RMSE={rmse:.4f}, α={alpha:.4f}, F1-macro={f1_m:.4f}, F1-weighted={f1_w:.4f}\n")
        f.write("\n")
        
        f.write("BREAKDOWN BY SUBSKILL:\n")
        f.write("-" * 40 + "\n\n")
        
        # Create detailed subskill table with standard deviations
        f.write("| Subskill | Accuracy±Std | RMSE±Std | α±Std | F1-macro±Std | F1-weighted±Std |\n")
        f.write("|----------|--------------|----------|-------|--------------|----------------|\n")
        
        subskill_order = ['Subskill 2.1', 'Subskill 2.2', 'Subskill 3.1', 'Subskill 3.2', 'Subskill 4.1', 'Subskill 4.2']
        
        for subskill in subskill_order:
            if subskill in all_subskill_metrics:
                metrics = all_subskill_metrics[subskill]
                subskill_name = subskill.replace("Subskill ", "")
                
                acc_mean = np.mean(metrics['accuracies'])
                acc_std = np.std(metrics['accuracies'])
                rmse_mean = np.mean(metrics['rmses'])
                rmse_std = np.std(metrics['rmses'])
                alpha_mean = np.mean(metrics['alphas'])
                alpha_std = np.std(metrics['alphas'])
                f1m_mean = np.mean(metrics['f1_macros'])
                f1m_std = np.std(metrics['f1_macros'])
                f1w_mean = np.mean(metrics['f1_weighteds'])
                f1w_std = np.std(metrics['f1_weighteds'])
                
                f.write(f"| {subskill_name:<8} | {acc_mean:.3f}±{acc_std:.3f}   | {rmse_mean:.3f}±{rmse_std:.3f} | {alpha_mean:.3f}±{alpha_std:.3f}   | {f1m_mean:.3f}±{f1m_std:.3f}   | {f1w_mean:.3f}±{f1w_std:.3f}    |\n")
        
        f.write("\n")
    
    # Also save CSV with detailed statistics
    csv_data = []
    for subskill in subskill_order:
        if subskill in all_subskill_metrics:
            metrics = all_subskill_metrics[subskill]
            csv_data.append({
                'Subskill': subskill.replace("Subskill ", ""),
                'Accuracy_Mean': np.mean(metrics['accuracies']),
                'Accuracy_Std': np.std(metrics['accuracies']),
                'RMSE_Mean': np.mean(metrics['rmses']),
                'RMSE_Std': np.std(metrics['rmses']),
                'Precision_Macro_Mean': np.mean(metrics['precision_macros']),
                'Precision_Macro_Std': np.std(metrics['precision_macros']),
                'Recall_Macro_Mean': np.mean(metrics['recall_macros']),
                'Recall_Macro_Std': np.std(metrics['recall_macros']),
                'F1_Macro_Mean': np.mean(metrics['f1_macros']),
                'F1_Macro_Std': np.std(metrics['f1_macros']),
                'Precision_Weighted_Mean': np.mean(metrics['precision_weighteds']),
                'Precision_Weighted_Std': np.std(metrics['precision_weighteds']),
                'Recall_Weighted_Mean': np.mean(metrics['recall_weighteds']),
                'Recall_Weighted_Std': np.std(metrics['recall_weighteds']),
                'F1_Weighted_Mean': np.mean(metrics['f1_weighteds']),
                'F1_Weighted_Std': np.std(metrics['f1_weighteds'])
            })
    
    # Add overall combined metrics to CSV
    csv_data.append({
        'Subskill': 'OVERALL_COMBINED',
        'Accuracy_Mean': mean_accuracy,
        'Accuracy_Std': std_accuracy,
        'RMSE_Mean': overall_rmse_combined,
        'RMSE_Std': 0,  # Single value, no std
        'Precision_Macro_Mean': overall_precision_macro,
        'Precision_Macro_Std': 0,
        'Recall_Macro_Mean': overall_recall_macro,
        'Recall_Macro_Std': 0,
        'F1_Macro_Mean': overall_f1_macro,
        'F1_Macro_Std': 0,
        'Precision_Weighted_Mean': overall_precision_weighted,
        'Precision_Weighted_Std': 0,
        'Recall_Weighted_Mean': overall_recall_weighted,
        'Recall_Weighted_Std': 0,
        'F1_Weighted_Mean': overall_f1_weighted,
        'F1_Weighted_Std': 0
    })
    
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(os.path.join(overall_results_dir, "subskill_statistics.csv"), index=False)
    
    # Save simple JSON file for statistical significance testing
    simple_results = {
        "model_type": f"{prefix}_{result_suffix}",
        "accuracies": [float(acc) for acc in all_accuracies],
        "rmses": [float(rmse) for rmse in all_rmses],
        "f1_macros": [float(f1_m) for f1_m in all_f1_macros],
        "f1_weighteds": [float(f1_w) for f1_w in all_f1_weighteds]
    }
    
    json_file = os.path.join(overall_results_dir, f"{prefix}_{result_suffix}_results.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(simple_results, f, indent=2)
    
    print(f"\nFT_Split results saved to: {ft_splits_dir}")
    print(f"  - Individual split results in: {split_dirs}")
    print(f"  - Overall results: {overall_results_dir}")
    print(f"    * overall_statistics.txt: Detailed report with standard deviations")
    print(f"    * subskill_statistics.csv: Machine-readable statistics")
    print(f"    * {prefix}_{result_suffix}_results.json: Simple JSON for significance testing")

def _traits_dir_suffix(traits_input=False, traits_output=False, trait_observations=False, obs_only=False, contrastive=False):
    """Build the directory suffix that matches llama_train_wrubric_traits naming."""
    if obs_only:
        obs = "_obs_only"
    elif trait_observations:
        obs = "_observations"
    else:
        obs = ""
    if traits_input and traits_output:
        return f"_traits_input_output{obs}"
    if traits_input:
        return f"_traits_input{obs}"
    if traits_output:
        return f"_traits_output{obs}"
    # Contrastive-only: traits used as training signal but not injected into prompt
    if contrastive:
        return "_traits_contrastive_only"
    return ""


def analyze_generalization_results(model_type="bert", wrubric=False,
                                   traits_input=False, traits_output=False,
                                   trait_observations=False, obs_only=False,
                                   contrastive=False):
    """
    Analyze results from generalization experiments
    UNIQUE_GENERALIZATION_FUNCTION
    """
    model_name = model_type.upper()
    if wrubric:
        model_name += "_WRUBRIC"

    traits_suffix = _traits_dir_suffix(traits_input, traits_output, trait_observations, obs_only, contrastive)
    if traits_suffix:
        model_name += traits_suffix.upper()

    print("=" * 80)
    print(f"ANALYZING {model_name} GENERALIZATION EXPERIMENT RESULTS")
    print("=" * 80)

    if model_type == "llama_no_just":
        base_dir = "llama_results_no_just_wrubric_generalize" if wrubric else "llama_results_no_just_generalize"
    elif model_type == "llama":
        base_dir = "llama_results_wrubric_generalize" if wrubric else "llama_results_generalize"
    elif model_type == "modernbert":
        base_dir = "modernbert_results_wrubric_generalize" if wrubric else "modernbert_results_generalize"
    else:
        base_dir = f"{model_type}_results_generalize"

    base_dir = base_dir + traits_suffix
    if contrastive:
        base_dir = base_dir + "_contrastive"

    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} not found!")
        print(f"Please run: python {model_type}_classify.py --generalize first")
        return
    
    # Load ground truth data
    gt_df = pd.read_excel("combined_essays.xlsx")
    gt_df.columns = gt_df.iloc[0]
    gt_df = gt_df[1:]
    
    # Define the experiments and their target subskills
    experiments = [
        {
            'name': 'subskill_4',
            'test_subskills': ['4.1', '4.2'],
            'description': 'Train on 2,3 -> Test on 4'
        },
        {
            'name': 'subskill_3', 
            'test_subskills': ['3.1', '3.2'],
            'description': 'Train on 2,4 -> Test on 3'
        },
        {
            'name': 'subskill_2',
            'test_subskills': ['2.1', '2.2'], 
            'description': 'Train on 3,4 -> Test on 2'
        }
    ]
    
    # Find experiment directories
    exp_dirs = []
    for exp in experiments:
        exp_dir = os.path.join(base_dir, exp['name'])
        if os.path.exists(exp_dir):
            exp_dirs.append((exp, exp_dir))
        else:
            print(f"Warning: Experiment directory {exp_dir} not found!")
    
    if not exp_dirs:
        print(f"Error: No experiment directories found in {base_dir}")
        return
    
    print(f"Found {len(exp_dirs)} experiment directories")
    
    # Create overall results directory
    overall_results_dir = os.path.join(base_dir, "overall_results")
    os.makedirs(overall_results_dir, exist_ok=True)
    
    all_results = {}
    combined_all_gt = []   # accumulates preds from all 3 held-out test sets
    combined_all_pred = []

    # Process each experiment
    for exp_config, exp_dir in exp_dirs:
        print(f"\n{'-'*60}")
        print(f"Processing: {exp_config['description']}")
        print(f"Directory: {exp_dir}")
        print(f"Target subskills: {exp_config['test_subskills']}")
        
        # Load test results
        test_results_file = os.path.join(exp_dir, "test_results.json")
        if not os.path.exists(test_results_file):
            print(f"Error: {test_results_file} not found!")
            continue
        
        with open(test_results_file, "r", encoding="utf-8") as f:
            pred_results = json.load(f)
        
        # Process results for this experiment (same logic as single model analysis)
        overall_accuracy, overall_rmse, overall_alpha, subskill_metrics = process_single_model_results(
            pred_results, gt_df, exp_dir
        )
        
        # Calculate overall F1 scores for this experiment
        # First, create the results dataframe again to get F1 scores
        results_data = []
        subskill_map = {
            '2.1': '2.1: Synthesizing Multiple Sources',
            '2.2': '2.2: Evaluating Evidence Strength',
            '3.1': '3.1: Using Counterarguments',
            '3.2': '3.2: Using Facts and Opinions',
            '4.1': '4.1: Drawing Conclusions',
            '4.2': '4.2: Using Logical Fallacies'
        }
        
        for _, row in gt_df.iterrows():
            essay_id = row['essay_id']
            if essay_id not in pred_results:
                continue
            pred_rubric = pred_results[essay_id]
            for subskill, gt_label in row.items():
                if 'Subskill' not in subskill:
                    continue
                subskill_number = subskill.split('Subskill ')[-1]
                subskill_name = subskill_map.get(subskill_number, subskill)
                pred_label = pred_rubric.get(subskill_name, {}).get('label', None)
                if pred_label not in ['Few-shot Example', 'Parsing Error'] and pred_label is not None:
                    # Only include test subskills for this experiment
                    if subskill_number in exp_config['test_subskills']:
                        results_data.append({
                            'Essay ID': essay_id,
                            'Subskill': subskill,
                            'Ground Truth': gt_label.lower().strip(),
                            'Predicted': pred_label.lower().strip()
                        })
        
        # Calculate overall F1 scores for this experiment
        if results_data:
            experiment_df = pd.DataFrame(results_data)
            from sklearn.metrics import classification_report
            
            # Get F1 scores from classification report
            exp_report_dict = classification_report(experiment_df['Ground Truth'], experiment_df['Predicted'], 
                                                   zero_division=0, output_dict=True)
            exp_macro = exp_report_dict.get('macro avg', {})
            exp_weighted = exp_report_dict.get('weighted avg', {})
            
            overall_f1_macro = exp_macro.get('f1-score', 0)
            overall_f1_weighted = exp_weighted.get('f1-score', 0)
            total_samples = len(experiment_df)

            # Accumulate for combined overall metrics
            combined_all_gt.extend(experiment_df['Ground Truth'].tolist())
            combined_all_pred.extend(experiment_df['Predicted'].tolist())
        else:
            overall_f1_macro = 0
            overall_f1_weighted = 0
            total_samples = 0
        
        # Filter subskill metrics to only include test subskills
        filtered_subskill_metrics = {}
        for subskill_key, metrics in subskill_metrics.items():
            subskill_num = subskill_key.replace("Subskill ", "")
            if subskill_num in exp_config['test_subskills']:
                filtered_subskill_metrics[subskill_key] = metrics
        
        # Store results for overall summary
        all_results[exp_config['name']] = {
            'description': exp_config['description'],
            'test_subskills': exp_config['test_subskills'],
            'overall_accuracy': overall_accuracy,
            'overall_alpha': overall_alpha,
            'overall_rmse': overall_rmse,
            'overall_f1_macro': overall_f1_macro,
            'overall_f1_weighted': overall_f1_weighted,
            'total_samples': total_samples,
            'subskill_metrics': filtered_subskill_metrics
        }
        
        print(f"✅ Results processed for {exp_config['name']}")
        print(f"   Overall Accuracy: {overall_accuracy:.4f}")
        print(f"   Overall RMSE: {overall_rmse:.4f}")
        print(f"   Overall Krippendorff's α: {overall_alpha:.4f}")
        print(f"   Overall F1-macro: {overall_f1_macro:.4f}")
        print(f"   Overall F1-weighted: {overall_f1_weighted:.4f}")
        print(f"   Total samples: {total_samples}")
        print(f"   Test subskills: {list(filtered_subskill_metrics.keys())}")
    
    # Compute combined overall metrics across all 6 held-out test subskills
    if combined_all_gt:
        from sklearn.metrics import classification_report as _cr
        combined_report = _cr(combined_all_gt, combined_all_pred, zero_division=0, output_dict=True)
        combined_accuracy = sum(g == p for g, p in zip(combined_all_gt, combined_all_pred)) / len(combined_all_gt)
        combined_rmse = calculate_rmse(combined_all_gt, combined_all_pred)
        combined_alpha = calculate_krippendorff_alpha(combined_all_gt, combined_all_pred)
        combined_f1_macro = combined_report.get('macro avg', {}).get('f1-score', 0)
        combined_f1_weighted = combined_report.get('weighted avg', {}).get('f1-score', 0)
        combined_precision_macro = combined_report.get('macro avg', {}).get('precision', 0)
        combined_recall_macro = combined_report.get('macro avg', {}).get('recall', 0)
        combined_precision_weighted = combined_report.get('weighted avg', {}).get('precision', 0)
        combined_recall_weighted = combined_report.get('weighted avg', {}).get('recall', 0)
        combined_total = len(combined_all_gt)
    else:
        combined_accuracy = combined_rmse = combined_alpha = 0
        combined_f1_macro = combined_f1_weighted = 0
        combined_precision_macro = combined_recall_macro = 0
        combined_precision_weighted = combined_recall_weighted = 0
        combined_total = 0

    # Create overall summary
    print(f"\n{'='*80}")
    print(f"{model_name} GENERALIZATION EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    # Save overall summary file
    summary_file = os.path.join(overall_results_dir, "generalization_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"{model_name} GENERALIZATION EXPERIMENT SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        for exp_name, results in all_results.items():
            print(f"\n{results['description']}:")
            print(f"  Overall Accuracy: {results['overall_accuracy']:.4f}")
            print(f"  Overall RMSE: {results['overall_rmse']:.4f}")
            print(f"  Overall Krippendorff's α: {results['overall_alpha']:.4f}")
            print(f"  Overall F1-macro: {results['overall_f1_macro']:.4f}")
            print(f"  Overall F1-weighted: {results['overall_f1_weighted']:.4f}")
            print(f"  Total samples: {results['total_samples']}")
            
            f.write(f"{results['description']}:\n")
            f.write(f"  Overall Accuracy: {results['overall_accuracy']:.4f}\n")
            f.write(f"  Overall RMSE: {results['overall_rmse']:.4f}\n")
            f.write(f"  Overall Krippendorff's α: {results['overall_alpha']:.4f}\n")
            f.write(f"  Overall F1-macro: {results['overall_f1_macro']:.4f}\n")
            f.write(f"  Overall F1-weighted: {results['overall_f1_weighted']:.4f}\n")
            f.write(f"  Total samples: {results['total_samples']}\n")
            
            # Show individual subskill results for test subskills only
            for subskill, metrics in results['subskill_metrics'].items():
                subskill_clean = subskill.replace("Subskill ", "")
                accuracy = metrics.get('accuracy', 0)
                rmse = metrics.get('rmse', 0)
                alpha = metrics.get('alpha', 0)
                total_samples = metrics.get('total_samples', 0)
                f1_macro = metrics.get('f1_macro', 0)
                f1_weighted = metrics.get('f1_weighted', 0)
                
                print(f"    {subskill_clean}: Acc={accuracy:.4f}, RMSE={rmse:.4f}, α={alpha:.4f}, F1-macro={f1_macro:.4f}, F1-weighted={f1_weighted:.4f}, N={total_samples}")
                f.write(f"    {subskill_clean}: Acc={accuracy:.4f}, RMSE={rmse:.4f}, α={alpha:.4f}, F1-macro={f1_macro:.4f}, F1-weighted={f1_weighted:.4f}, N={total_samples}\n")
            
            f.write("\n")

        # Write combined overall metrics
        f.write("COMBINED OVERALL (all 6 held-out test subskills linearized):\n")
        f.write("=" * 80 + "\n")
        f.write(f"  Overall Accuracy:           {combined_accuracy:.4f}\n")
        f.write(f"  Overall RMSE:               {combined_rmse:.4f}\n")
        f.write(f"  Overall Krippendorff's α:   {combined_alpha:.4f}\n")
        f.write(f"  Overall F1-macro:           {combined_f1_macro:.4f}\n")
        f.write(f"  Overall F1-weighted:        {combined_f1_weighted:.4f}\n")
        f.write(f"  Precision (macro):          {combined_precision_macro:.4f}\n")
        f.write(f"  Recall (macro):             {combined_recall_macro:.4f}\n")
        f.write(f"  Precision (weighted):       {combined_precision_weighted:.4f}\n")
        f.write(f"  Recall (weighted):          {combined_recall_weighted:.4f}\n")
        f.write(f"  Total samples:              {combined_total}\n")

    # Print combined overall metrics
    print(f"\nCOMBINED OVERALL (all 6 held-out test subskills linearized):")
    print(f"  Overall Accuracy:           {combined_accuracy:.4f}")
    print(f"  Overall RMSE:               {combined_rmse:.4f}")
    print(f"  Overall Krippendorff's α:   {combined_alpha:.4f}")
    print(f"  Overall F1-macro:           {combined_f1_macro:.4f}")
    print(f"  Overall F1-weighted:        {combined_f1_weighted:.4f}")
    print(f"  Precision (macro):          {combined_precision_macro:.4f}")
    print(f"  Recall (macro):             {combined_recall_macro:.4f}")
    print(f"  Precision (weighted):       {combined_precision_weighted:.4f}")
    print(f"  Recall (weighted):          {combined_recall_weighted:.4f}")
    print(f"  Total samples:              {combined_total}")

    # Create CSV summary for easy analysis
    summary_data = []
    for exp_name, results in all_results.items():
        summary_data.append({
            'Experiment': exp_name,
            'Description': results['description'],
            'Test_Subskills': ', '.join(results['test_subskills']),
            'Overall_Accuracy': round(results['overall_accuracy'], 4),
            'Overall_RMSE': round(results['overall_rmse'], 4),
            'Overall_Alpha': round(results['overall_alpha'], 4),
            'F1_Macro': round(results['overall_f1_macro'], 4),
            'F1_Weighted': round(results['overall_f1_weighted'], 4),
            'Total_Samples': results['total_samples']
        })
        
        # Add individual subskill rows
        for subskill, metrics in results['subskill_metrics'].items():
            subskill_clean = subskill.replace("Subskill ", "")
            summary_data.append({
                'Experiment': f"{exp_name}_{subskill_clean}",
                'Description': f"{results['description']} - {subskill_clean}",
                'Test_Subskills': subskill_clean,
                'Overall_Accuracy': round(metrics.get('accuracy', 0), 4),
                'Overall_RMSE': round(metrics.get('rmse', 0), 4),
                'Overall_Alpha': round(metrics.get('alpha', 0), 4),
                'F1_Macro': round(metrics.get('f1_macro', 0), 4),
                'F1_Weighted': round(metrics.get('f1_weighted', 0), 4),
                'Total_Samples': metrics.get('total_samples', 0)
            })
    
    # Add combined overall row to CSV
    summary_data.append({
        'Experiment': 'OVERALL_COMBINED',
        'Description': 'All 6 held-out test subskills linearized',
        'Test_Subskills': '2.1, 2.2, 3.1, 3.2, 4.1, 4.2',
        'Overall_Accuracy': round(combined_accuracy, 4),
        'Overall_RMSE': round(combined_rmse, 4),
        'Overall_Alpha': round(combined_alpha, 4),
        'F1_Macro': round(combined_f1_macro, 4),
        'F1_Weighted': round(combined_f1_weighted, 4),
        'Total_Samples': combined_total
    })

    summary_df = pd.DataFrame(summary_data)
    # Round all float columns to 4 decimal places before saving
    float_columns = summary_df.select_dtypes(include=[np.float64, np.float32]).columns
    summary_df[float_columns] = summary_df[float_columns].round(4)
    summary_csv = os.path.join(overall_results_dir, "generalization_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    
    print(f"\nGeneralization results saved to: {base_dir}/overall_results/")
    print(f"  - generalization_summary.txt: Detailed summary")
    print(f"  - generalization_summary.csv: Machine-readable summary")
    
    print(f"\nIndividual experiment results available in:")
    for exp_config, exp_dir in exp_dirs:
        print(f"  - {exp_dir}/ ({exp_config['description']})")

def analyze_gen_train_results(model_type="bert", wrubric=False,
                              traits_input=False, traits_output=False,
                              trait_observations=False, obs_only=False,
                              contrastive=False):
    """
    Analyze results from --gen_train inference (train+val 80% essays, held-out subskill).
    Reads test_results_train.json from each experiment directory and writes
    overall_results_gen_train/generalization_summary.csv in the same format as
    analyze_generalization_results().
    UNIQUE_GEN_TRAIN_FUNCTION
    """
    model_name = model_type.upper()
    if wrubric:
        model_name += "_WRUBRIC"

    traits_suffix = _traits_dir_suffix(traits_input, traits_output, trait_observations, obs_only, contrastive)
    if traits_suffix:
        model_name += traits_suffix.upper()

    print("=" * 80)
    print(f"ANALYZING {model_name} GEN_TRAIN RESULTS (train+val essays, held-out subskill)")
    print("=" * 80)

    if model_type == "llama_no_just":
        base_dir = "llama_results_no_just_wrubric_generalize" if wrubric else "llama_results_no_just_generalize"
    elif model_type == "llama":
        base_dir = "llama_results_wrubric_generalize" if wrubric else "llama_results_generalize"
    elif model_type == "modernbert":
        base_dir = "modernbert_results_wrubric_generalize" if wrubric else "modernbert_results_generalize"
    else:
        base_dir = f"{model_type}_results_generalize"

    base_dir = base_dir + traits_suffix
    if contrastive:
        base_dir = base_dir + "_contrastive"

    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} not found!")
        print(f"Please run: python {model_type}_train.py --generalize first, then --gen_train")
        return

    # Load ground truth data
    gt_df = pd.read_excel("combined_essays.xlsx")
    gt_df.columns = gt_df.iloc[0]
    gt_df = gt_df[1:]

    # Define the experiments and their target subskills
    experiments = [
        {
            'name': 'subskill_4',
            'test_subskills': ['4.1', '4.2'],
            'description': 'Train on 2,3 -> Test on 4'
        },
        {
            'name': 'subskill_3',
            'test_subskills': ['3.1', '3.2'],
            'description': 'Train on 2,4 -> Test on 3'
        },
        {
            'name': 'subskill_2',
            'test_subskills': ['2.1', '2.2'],
            'description': 'Train on 3,4 -> Test on 2'
        }
    ]

    # Find experiment directories
    exp_dirs = []
    for exp in experiments:
        exp_dir = os.path.join(base_dir, exp['name'])
        if os.path.exists(exp_dir):
            exp_dirs.append((exp, exp_dir))
        else:
            print(f"Warning: Experiment directory {exp_dir} not found!")

    if not exp_dirs:
        print(f"Error: No experiment directories found in {base_dir}")
        return

    print(f"Found {len(exp_dirs)} experiment directories")

    # Create overall_results_gen_train directory (separate from overall_results)
    overall_results_dir = os.path.join(base_dir, "overall_results_gen_train")
    os.makedirs(overall_results_dir, exist_ok=True)

    all_results = {}
    combined_all_gt = []
    combined_all_pred = []

    # Process each experiment
    for exp_config, exp_dir in exp_dirs:
        print(f"\n{'-'*60}")
        print(f"Processing: {exp_config['description']}")
        print(f"Directory: {exp_dir}")
        print(f"Target subskills: {exp_config['test_subskills']}")

        # Load gen_train results (train+val essays)
        test_results_file = os.path.join(exp_dir, "test_results_train.json")
        if not os.path.exists(test_results_file):
            print(f"Error: {test_results_file} not found!")
            continue

        with open(test_results_file, "r", encoding="utf-8") as f:
            pred_results = json.load(f)

        # Process results for this experiment
        overall_accuracy, overall_rmse, overall_alpha, subskill_metrics = process_single_model_results(
            pred_results, gt_df, exp_dir
        )

        # Calculate overall F1 scores for this experiment
        results_data = []
        subskill_map = {
            '2.1': '2.1: Synthesizing Multiple Sources',
            '2.2': '2.2: Evaluating Evidence Strength',
            '3.1': '3.1: Using Counterarguments',
            '3.2': '3.2: Using Facts and Opinions',
            '4.1': '4.1: Drawing Conclusions',
            '4.2': '4.2: Using Logical Fallacies'
        }

        for _, row in gt_df.iterrows():
            essay_id = row['essay_id']
            if essay_id not in pred_results:
                continue
            pred_rubric = pred_results[essay_id]
            for subskill, gt_label in row.items():
                if 'Subskill' not in subskill:
                    continue
                subskill_number = subskill.split('Subskill ')[-1]
                subskill_name = subskill_map.get(subskill_number, subskill)
                pred_label = pred_rubric.get(subskill_name, {}).get('label', None)
                if pred_label not in ['Few-shot Example', 'Parsing Error'] and pred_label is not None:
                    if subskill_number in exp_config['test_subskills']:
                        results_data.append({
                            'Essay ID': essay_id,
                            'Subskill': subskill,
                            'Ground Truth': gt_label.lower().strip(),
                            'Predicted': pred_label.lower().strip()
                        })

        if results_data:
            experiment_df = pd.DataFrame(results_data)
            from sklearn.metrics import classification_report

            exp_report_dict = classification_report(experiment_df['Ground Truth'], experiment_df['Predicted'],
                                                    zero_division=0, output_dict=True)
            exp_macro = exp_report_dict.get('macro avg', {})
            exp_weighted = exp_report_dict.get('weighted avg', {})

            overall_f1_macro = exp_macro.get('f1-score', 0)
            overall_f1_weighted = exp_weighted.get('f1-score', 0)
            total_samples = len(experiment_df)

            combined_all_gt.extend(experiment_df['Ground Truth'].tolist())
            combined_all_pred.extend(experiment_df['Predicted'].tolist())
        else:
            overall_f1_macro = 0
            overall_f1_weighted = 0
            total_samples = 0

        # Filter subskill metrics to only include test subskills
        filtered_subskill_metrics = {}
        for subskill_key, metrics in subskill_metrics.items():
            subskill_num = subskill_key.replace("Subskill ", "")
            if subskill_num in exp_config['test_subskills']:
                filtered_subskill_metrics[subskill_key] = metrics

        all_results[exp_config['name']] = {
            'description': exp_config['description'],
            'test_subskills': exp_config['test_subskills'],
            'overall_accuracy': overall_accuracy,
            'overall_alpha': overall_alpha,
            'overall_rmse': overall_rmse,
            'overall_f1_macro': overall_f1_macro,
            'overall_f1_weighted': overall_f1_weighted,
            'total_samples': total_samples,
            'subskill_metrics': filtered_subskill_metrics
        }

        print(f"✅ Results processed for {exp_config['name']}")
        print(f"   Overall Accuracy: {overall_accuracy:.4f}")
        print(f"   Overall RMSE: {overall_rmse:.4f}")
        print(f"   Overall Krippendorff's α: {overall_alpha:.4f}")
        print(f"   Overall F1-macro: {overall_f1_macro:.4f}")
        print(f"   Overall F1-weighted: {overall_f1_weighted:.4f}")
        print(f"   Total samples: {total_samples}")
        print(f"   Test subskills: {list(filtered_subskill_metrics.keys())}")

    # Compute combined overall metrics across all 6 held-out subskills (on train+val essays)
    if combined_all_gt:
        from sklearn.metrics import classification_report as _cr
        combined_report = _cr(combined_all_gt, combined_all_pred, zero_division=0, output_dict=True)
        combined_accuracy = sum(g == p for g, p in zip(combined_all_gt, combined_all_pred)) / len(combined_all_gt)
        combined_rmse = calculate_rmse(combined_all_gt, combined_all_pred)
        combined_alpha = calculate_krippendorff_alpha(combined_all_gt, combined_all_pred)
        combined_f1_macro = combined_report.get('macro avg', {}).get('f1-score', 0)
        combined_f1_weighted = combined_report.get('weighted avg', {}).get('f1-score', 0)
        combined_precision_macro = combined_report.get('macro avg', {}).get('precision', 0)
        combined_recall_macro = combined_report.get('macro avg', {}).get('recall', 0)
        combined_precision_weighted = combined_report.get('weighted avg', {}).get('precision', 0)
        combined_recall_weighted = combined_report.get('weighted avg', {}).get('recall', 0)
        combined_total = len(combined_all_gt)
    else:
        combined_accuracy = combined_rmse = combined_alpha = 0
        combined_f1_macro = combined_f1_weighted = 0
        combined_precision_macro = combined_recall_macro = 0
        combined_precision_weighted = combined_recall_weighted = 0
        combined_total = 0

    print(f"\n{'='*80}")
    print(f"{model_name} GEN_TRAIN EXPERIMENT SUMMARY")
    print(f"{'='*80}")

    # Save summary .txt
    summary_file = os.path.join(overall_results_dir, "generalization_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"{model_name} GEN_TRAIN EXPERIMENT SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        for exp_name, results in all_results.items():
            print(f"\n{results['description']}:")
            print(f"  Overall Accuracy: {results['overall_accuracy']:.4f}")
            print(f"  Overall RMSE: {results['overall_rmse']:.4f}")
            print(f"  Overall Krippendorff's α: {results['overall_alpha']:.4f}")
            print(f"  Overall F1-macro: {results['overall_f1_macro']:.4f}")
            print(f"  Overall F1-weighted: {results['overall_f1_weighted']:.4f}")
            print(f"  Total samples: {results['total_samples']}")

            f.write(f"{results['description']}:\n")
            f.write(f"  Overall Accuracy: {results['overall_accuracy']:.4f}\n")
            f.write(f"  Overall RMSE: {results['overall_rmse']:.4f}\n")
            f.write(f"  Overall Krippendorff's α: {results['overall_alpha']:.4f}\n")
            f.write(f"  Overall F1-macro: {results['overall_f1_macro']:.4f}\n")
            f.write(f"  Overall F1-weighted: {results['overall_f1_weighted']:.4f}\n")
            f.write(f"  Total samples: {results['total_samples']}\n")

            for subskill, metrics in results['subskill_metrics'].items():
                subskill_clean = subskill.replace("Subskill ", "")
                accuracy = metrics.get('accuracy', 0)
                rmse = metrics.get('rmse', 0)
                alpha = metrics.get('alpha', 0)
                total_samples = metrics.get('total_samples', 0)
                f1_macro = metrics.get('f1_macro', 0)
                f1_weighted = metrics.get('f1_weighted', 0)

                print(f"    {subskill_clean}: Acc={accuracy:.4f}, RMSE={rmse:.4f}, α={alpha:.4f}, F1-macro={f1_macro:.4f}, F1-weighted={f1_weighted:.4f}, N={total_samples}")
                f.write(f"    {subskill_clean}: Acc={accuracy:.4f}, RMSE={rmse:.4f}, α={alpha:.4f}, F1-macro={f1_macro:.4f}, F1-weighted={f1_weighted:.4f}, N={total_samples}\n")

            f.write("\n")

        f.write("COMBINED OVERALL (all 6 held-out subskills on train+val essays):\n")
        f.write("=" * 80 + "\n")
        f.write(f"  Overall Accuracy:           {combined_accuracy:.4f}\n")
        f.write(f"  Overall RMSE:               {combined_rmse:.4f}\n")
        f.write(f"  Overall Krippendorff's α:   {combined_alpha:.4f}\n")
        f.write(f"  Overall F1-macro:           {combined_f1_macro:.4f}\n")
        f.write(f"  Overall F1-weighted:        {combined_f1_weighted:.4f}\n")
        f.write(f"  Precision (macro):          {combined_precision_macro:.4f}\n")
        f.write(f"  Recall (macro):             {combined_recall_macro:.4f}\n")
        f.write(f"  Precision (weighted):       {combined_precision_weighted:.4f}\n")
        f.write(f"  Recall (weighted):          {combined_recall_weighted:.4f}\n")
        f.write(f"  Total samples:              {combined_total}\n")

    print(f"\nCOMBINED OVERALL (all 6 held-out subskills on train+val essays):")
    print(f"  Overall Accuracy:           {combined_accuracy:.4f}")
    print(f"  Overall RMSE:               {combined_rmse:.4f}")
    print(f"  Overall Krippendorff's α:   {combined_alpha:.4f}")
    print(f"  Overall F1-macro:           {combined_f1_macro:.4f}")
    print(f"  Overall F1-weighted:        {combined_f1_weighted:.4f}")
    print(f"  Precision (macro):          {combined_precision_macro:.4f}")
    print(f"  Recall (macro):             {combined_recall_macro:.4f}")
    print(f"  Precision (weighted):       {combined_precision_weighted:.4f}")
    print(f"  Recall (weighted):          {combined_recall_weighted:.4f}")
    print(f"  Total samples:              {combined_total}")

    # Build CSV summary
    summary_data = []
    for exp_name, results in all_results.items():
        summary_data.append({
            'Experiment': exp_name,
            'Description': results['description'],
            'Test_Subskills': ', '.join(results['test_subskills']),
            'Overall_Accuracy': round(results['overall_accuracy'], 4),
            'Overall_RMSE': round(results['overall_rmse'], 4),
            'Overall_Alpha': round(results['overall_alpha'], 4),
            'F1_Macro': round(results['overall_f1_macro'], 4),
            'F1_Weighted': round(results['overall_f1_weighted'], 4),
            'Total_Samples': results['total_samples']
        })

        for subskill, metrics in results['subskill_metrics'].items():
            subskill_clean = subskill.replace("Subskill ", "")
            summary_data.append({
                'Experiment': f"{exp_name}_{subskill_clean}",
                'Description': f"{results['description']} - {subskill_clean}",
                'Test_Subskills': subskill_clean,
                'Overall_Accuracy': round(metrics.get('accuracy', 0), 4),
                'Overall_RMSE': round(metrics.get('rmse', 0), 4),
                'Overall_Alpha': round(metrics.get('alpha', 0), 4),
                'F1_Macro': round(metrics.get('f1_macro', 0), 4),
                'F1_Weighted': round(metrics.get('f1_weighted', 0), 4),
                'Total_Samples': metrics.get('total_samples', 0)
            })

    summary_data.append({
        'Experiment': 'OVERALL_COMBINED',
        'Description': 'All 6 held-out subskills on train+val essays',
        'Test_Subskills': '2.1, 2.2, 3.1, 3.2, 4.1, 4.2',
        'Overall_Accuracy': round(combined_accuracy, 4),
        'Overall_RMSE': round(combined_rmse, 4),
        'Overall_Alpha': round(combined_alpha, 4),
        'F1_Macro': round(combined_f1_macro, 4),
        'F1_Weighted': round(combined_f1_weighted, 4),
        'Total_Samples': combined_total
    })

    summary_df = pd.DataFrame(summary_data)
    float_columns = summary_df.select_dtypes(include=[np.float64, np.float32]).columns
    summary_df[float_columns] = summary_df[float_columns].round(4)
    summary_csv = os.path.join(overall_results_dir, "generalization_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print(f"\nGen-train results saved to: {base_dir}/overall_results_gen_train/")
    print(f"  - generalization_summary.txt: Detailed summary")
    print(f"  - generalization_summary.csv: Machine-readable summary")

    print(f"\nIndividual experiment results available in:")
    for exp_config, exp_dir in exp_dirs:
        print(f"  - {exp_dir}/ ({exp_config['description']})")


def analyze_generalize_seen_results(model_type="llama", wrubric=False,
                                    traits_input=False, traits_output=False,
                                    trait_observations=False, obs_only=False,
                                    contrastive=False):
    """
    Analyze results from --generalize_seen experiments.
    Reads both test_results.json (20% test essays) and test_results_train.json
    (80% train+val essays) from each <base>_generalize_seen/subskill_X/ directory
    and writes two summary sets to overall_results_generalize_seen/.
    UNIQUE_GENERALIZE_SEEN_FUNCTION
    """
    model_name = model_type.upper()
    if wrubric:
        model_name += "_WRUBRIC"

    traits_suffix = _traits_dir_suffix(traits_input, traits_output, trait_observations, obs_only, contrastive)
    if traits_suffix:
        model_name += traits_suffix.upper()

    print("=" * 80)
    print(f"ANALYZING {model_name} GENERALIZE_SEEN RESULTS")
    print("  test_results.json      = 20% test essays, unseen subskill")
    print("  test_results_train.json = 80% train+val essays, unseen subskill")
    print("=" * 80)

    if model_type == "llama_no_just":
        base_dir = "llama_results_no_just_wrubric_generalize_seen" if wrubric else "llama_results_no_just_generalize_seen"
    elif model_type == "llama":
        base_dir = "llama_results_wrubric_generalize_seen" if wrubric else "llama_results_generalize_seen"
    elif model_type == "modernbert":
        base_dir = "modernbert_results_wrubric_generalize_seen" if wrubric else "modernbert_results_generalize_seen"
    else:
        base_dir = f"{model_type}_results_generalize_seen"

    base_dir = base_dir + traits_suffix
    if contrastive:
        base_dir = base_dir + "_contrastive"

    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} not found!")
        print(f"Please run: python {model_type}_train_wrubric.py --generalize_seen first")
        return

    # Load ground truth data
    gt_df = pd.read_excel("combined_essays.xlsx")
    gt_df.columns = gt_df.iloc[0]
    gt_df = gt_df[1:]

    experiments = [
        {'name': 'subskill_4', 'test_subskills': ['4.1', '4.2'], 'description': 'Train on 2,3 -> Test on 4'},
        {'name': 'subskill_3', 'test_subskills': ['3.1', '3.2'], 'description': 'Train on 2,4 -> Test on 3'},
        {'name': 'subskill_2', 'test_subskills': ['2.1', '2.2'], 'description': 'Train on 3,4 -> Test on 2'},
    ]

    exp_dirs = []
    for exp in experiments:
        exp_dir = os.path.join(base_dir, exp['name'])
        if os.path.exists(exp_dir):
            exp_dirs.append((exp, exp_dir))
        else:
            print(f"Warning: Experiment directory {exp_dir} not found!")

    if not exp_dirs:
        print(f"Error: No experiment directories found in {base_dir}")
        return

    print(f"Found {len(exp_dirs)} experiment directories")

    overall_results_dir = os.path.join(base_dir, "overall_results_generalize_seen")
    os.makedirs(overall_results_dir, exist_ok=True)

    subskill_map = {
        '2.1': '2.1: Synthesizing Multiple Sources',
        '2.2': '2.2: Evaluating Evidence Strength',
        '3.1': '3.1: Using Counterarguments',
        '3.2': '3.2: Using Facts and Opinions',
        '4.1': '4.1: Drawing Conclusions',
        '4.2': '4.2: Using Logical Fallacies'
    }

    def _run_analysis_for_split(results_filename, split_label):
        """Inner helper: analyse one results file (test or train) across all experiment dirs."""
        all_results = {}
        combined_all_gt = []
        combined_all_pred = []

        for exp_config, exp_dir in exp_dirs:
            print(f"\n{'-'*60}")
            print(f"[{split_label}] Processing: {exp_config['description']}")
            print(f"Directory: {exp_dir}")
            print(f"Target subskills: {exp_config['test_subskills']}")

            results_file = os.path.join(exp_dir, results_filename)
            if not os.path.exists(results_file):
                print(f"Error: {results_file} not found! Skipping.")
                continue

            with open(results_file, "r", encoding="utf-8") as f:
                pred_results = json.load(f)

            overall_accuracy, overall_rmse, overall_alpha, subskill_metrics = process_single_model_results(
                pred_results, gt_df, exp_dir
            )

            results_data = []
            for _, row in gt_df.iterrows():
                essay_id = row['essay_id']
                if essay_id not in pred_results:
                    continue
                pred_rubric = pred_results[essay_id]
                for subskill, gt_label in row.items():
                    if 'Subskill' not in subskill:
                        continue
                    subskill_number = subskill.split('Subskill ')[-1]
                    subskill_name = subskill_map.get(subskill_number, subskill)
                    pred_label = pred_rubric.get(subskill_name, {}).get('label', None)
                    if pred_label not in ['Few-shot Example', 'Parsing Error'] and pred_label is not None:
                        if subskill_number in exp_config['test_subskills']:
                            results_data.append({
                                'Essay ID': essay_id,
                                'Subskill': subskill,
                                'Ground Truth': gt_label.lower().strip(),
                                'Predicted': pred_label.lower().strip()
                            })

            if results_data:
                experiment_df = pd.DataFrame(results_data)
                from sklearn.metrics import classification_report
                exp_report_dict = classification_report(experiment_df['Ground Truth'], experiment_df['Predicted'],
                                                        zero_division=0, output_dict=True)
                overall_f1_macro = exp_report_dict.get('macro avg', {}).get('f1-score', 0)
                overall_f1_weighted = exp_report_dict.get('weighted avg', {}).get('f1-score', 0)
                total_samples = len(experiment_df)
                combined_all_gt.extend(experiment_df['Ground Truth'].tolist())
                combined_all_pred.extend(experiment_df['Predicted'].tolist())
            else:
                overall_f1_macro = overall_f1_weighted = total_samples = 0

            filtered_subskill_metrics = {
                k: v for k, v in subskill_metrics.items()
                if k.replace('Subskill ', '') in exp_config['test_subskills']
            }

            all_results[exp_config['name']] = {
                'description': exp_config['description'],
                'test_subskills': exp_config['test_subskills'],
                'overall_accuracy': overall_accuracy,
                'overall_alpha': overall_alpha,
                'overall_rmse': overall_rmse,
                'overall_f1_macro': overall_f1_macro,
                'overall_f1_weighted': overall_f1_weighted,
                'total_samples': total_samples,
                'subskill_metrics': filtered_subskill_metrics
            }

            print(f"\u2705 Results processed for {exp_config['name']}")
            print(f"   Overall Accuracy: {overall_accuracy:.4f}")
            print(f"   Overall RMSE: {overall_rmse:.4f}")
            print(f"   Overall Krippendorff's \u03b1: {overall_alpha:.4f}")
            print(f"   Overall F1-macro: {overall_f1_macro:.4f}")
            print(f"   Overall F1-weighted: {overall_f1_weighted:.4f}")
            print(f"   Total samples: {total_samples}")

        # Combined metrics
        if combined_all_gt:
            from sklearn.metrics import classification_report as _cr
            combined_report = _cr(combined_all_gt, combined_all_pred, zero_division=0, output_dict=True)
            combined_accuracy = sum(g == p for g, p in zip(combined_all_gt, combined_all_pred)) / len(combined_all_gt)
            combined_rmse = calculate_rmse(combined_all_gt, combined_all_pred)
            combined_alpha = calculate_krippendorff_alpha(combined_all_gt, combined_all_pred)
            combined_f1_macro = combined_report.get('macro avg', {}).get('f1-score', 0)
            combined_f1_weighted = combined_report.get('weighted avg', {}).get('f1-score', 0)
            combined_precision_macro = combined_report.get('macro avg', {}).get('precision', 0)
            combined_recall_macro = combined_report.get('macro avg', {}).get('recall', 0)
            combined_precision_weighted = combined_report.get('weighted avg', {}).get('precision', 0)
            combined_recall_weighted = combined_report.get('weighted avg', {}).get('recall', 0)
            combined_total = len(combined_all_gt)
        else:
            combined_accuracy = combined_rmse = combined_alpha = 0
            combined_f1_macro = combined_f1_weighted = 0
            combined_precision_macro = combined_recall_macro = 0
            combined_precision_weighted = combined_recall_weighted = 0
            combined_total = 0

        print(f"\n{'='*80}")
        print(f"{model_name} GENERALIZE_SEEN [{split_label.upper()}] SUMMARY")
        print(f"{'='*80}")

        # Derive file stem from split (test_results.json -> test, test_results_train.json -> train)
        file_stem = "train" if "train" in results_filename else "test"
        summary_txt = os.path.join(overall_results_dir, f"generalization_summary_{file_stem}.txt")
        summary_csv_path = os.path.join(overall_results_dir, f"generalization_summary_{file_stem}.csv")

        with open(summary_txt, 'w', encoding='utf-8') as f:
            f.write(f"{model_name} GENERALIZE_SEEN [{split_label.upper()}] SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            for exp_name, results in all_results.items():
                print(f"\n{results['description']}:")
                print(f"  Overall Accuracy: {results['overall_accuracy']:.4f}")
                print(f"  Overall RMSE: {results['overall_rmse']:.4f}")
                print(f"  Overall Krippendorff's \u03b1: {results['overall_alpha']:.4f}")
                print(f"  Overall F1-macro: {results['overall_f1_macro']:.4f}")
                print(f"  Overall F1-weighted: {results['overall_f1_weighted']:.4f}")
                print(f"  Total samples: {results['total_samples']}")

                f.write(f"{results['description']}:\n")
                f.write(f"  Overall Accuracy: {results['overall_accuracy']:.4f}\n")
                f.write(f"  Overall RMSE: {results['overall_rmse']:.4f}\n")
                f.write(f"  Overall Krippendorff's \u03b1: {results['overall_alpha']:.4f}\n")
                f.write(f"  Overall F1-macro: {results['overall_f1_macro']:.4f}\n")
                f.write(f"  Overall F1-weighted: {results['overall_f1_weighted']:.4f}\n")
                f.write(f"  Total samples: {results['total_samples']}\n")

                for subskill, metrics in results['subskill_metrics'].items():
                    sk = subskill.replace('Subskill ', '')
                    acc = metrics.get('accuracy', 0)
                    rmse = metrics.get('rmse', 0)
                    alpha = metrics.get('alpha', 0)
                    n = metrics.get('total_samples', 0)
                    f1m = metrics.get('f1_macro', 0)
                    f1w = metrics.get('f1_weighted', 0)
                    print(f"    {sk}: Acc={acc:.4f}, RMSE={rmse:.4f}, \u03b1={alpha:.4f}, F1-macro={f1m:.4f}, F1-weighted={f1w:.4f}, N={n}")
                    f.write(f"    {sk}: Acc={acc:.4f}, RMSE={rmse:.4f}, \u03b1={alpha:.4f}, F1-macro={f1m:.4f}, F1-weighted={f1w:.4f}, N={n}\n")
                f.write("\n")

            label_str = "all 6 held-out subskills on train+val essays" if file_stem == "train" else "all 6 held-out test subskills"
            f.write(f"COMBINED OVERALL ({label_str}):\n")
            f.write("=" * 80 + "\n")
            f.write(f"  Overall Accuracy:           {combined_accuracy:.4f}\n")
            f.write(f"  Overall RMSE:               {combined_rmse:.4f}\n")
            f.write(f"  Overall Krippendorff's \u03b1:   {combined_alpha:.4f}\n")
            f.write(f"  Overall F1-macro:           {combined_f1_macro:.4f}\n")
            f.write(f"  Overall F1-weighted:        {combined_f1_weighted:.4f}\n")
            f.write(f"  Precision (macro):          {combined_precision_macro:.4f}\n")
            f.write(f"  Recall (macro):             {combined_recall_macro:.4f}\n")
            f.write(f"  Precision (weighted):       {combined_precision_weighted:.4f}\n")
            f.write(f"  Recall (weighted):          {combined_recall_weighted:.4f}\n")
            f.write(f"  Total samples:              {combined_total}\n")

        print(f"\nCOMBINED OVERALL:")
        print(f"  Overall Accuracy:           {combined_accuracy:.4f}")
        print(f"  Overall RMSE:               {combined_rmse:.4f}")
        print(f"  Overall Krippendorff's \u03b1:   {combined_alpha:.4f}")
        print(f"  Overall F1-macro:           {combined_f1_macro:.4f}")
        print(f"  Overall F1-weighted:        {combined_f1_weighted:.4f}")
        print(f"  Total samples:              {combined_total}")

        # CSV
        summary_data = []
        for exp_name, results in all_results.items():
            summary_data.append({
                'Experiment': exp_name,
                'Description': results['description'],
                'Test_Subskills': ', '.join(results['test_subskills']),
                'Overall_Accuracy': round(results['overall_accuracy'], 4),
                'Overall_RMSE': round(results['overall_rmse'], 4),
                'Overall_Alpha': round(results['overall_alpha'], 4),
                'F1_Macro': round(results['overall_f1_macro'], 4),
                'F1_Weighted': round(results['overall_f1_weighted'], 4),
                'Total_Samples': results['total_samples']
            })
            for subskill, metrics in results['subskill_metrics'].items():
                sk = subskill.replace('Subskill ', '')
                summary_data.append({
                    'Experiment': f"{exp_name}_{sk}",
                    'Description': f"{results['description']} - {sk}",
                    'Test_Subskills': sk,
                    'Overall_Accuracy': round(metrics.get('accuracy', 0), 4),
                    'Overall_RMSE': round(metrics.get('rmse', 0), 4),
                    'Overall_Alpha': round(metrics.get('alpha', 0), 4),
                    'F1_Macro': round(metrics.get('f1_macro', 0), 4),
                    'F1_Weighted': round(metrics.get('f1_weighted', 0), 4),
                    'Total_Samples': metrics.get('total_samples', 0)
                })

        summary_data.append({
            'Experiment': 'OVERALL_COMBINED',
            'Description': label_str,
            'Test_Subskills': '2.1, 2.2, 3.1, 3.2, 4.1, 4.2',
            'Overall_Accuracy': round(combined_accuracy, 4),
            'Overall_RMSE': round(combined_rmse, 4),
            'Overall_Alpha': round(combined_alpha, 4),
            'F1_Macro': round(combined_f1_macro, 4),
            'F1_Weighted': round(combined_f1_weighted, 4),
            'Total_Samples': combined_total
        })

        summary_df = pd.DataFrame(summary_data)
        float_cols = summary_df.select_dtypes(include=[np.float64, np.float32]).columns
        summary_df[float_cols] = summary_df[float_cols].round(4)
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"  Saved: {summary_csv_path}")

    # Run analysis for both splits
    print(f"\n{'#'*80}")
    print("# SPLIT 1: 20% TEST ESSAYS (test_results.json)")
    print(f"{'#'*80}")
    _run_analysis_for_split("test_results.json", "test essays")

    print(f"\n{'#'*80}")
    print("# SPLIT 2: 80% TRAIN+VAL ESSAYS (test_results_train.json)")
    print(f"{'#'*80}")
    _run_analysis_for_split("test_results_train.json", "train+val essays")

    print(f"\nGeneralize-seen results saved to: {overall_results_dir}/")
    print(f"  - generalization_summary_test.txt / .csv")
    print(f"  - generalization_summary_train.txt / .csv")


def calculate_rmse(ground_truth, predicted):
    """
    Calculate RMSE for categorical labels mapped to integers
    
    Label mapping:
    not applicable - 0
    below emerging - 1
    emerging - 2
    expanding - 3
    exemplifying - 4
    
    Args:
        ground_truth: Series or list of ground truth labels
        predicted: Series or list of predicted labels
        
    Returns:
        float: RMSE value
    """
    label_to_int = {
        'not applicable': 0,
        'below emerging': 1,
        'emerging': 2,
        'expanding': 3,
        'exemplifying': 4
    }
    
    # Convert labels to integers
    gt_integers = [label_to_int.get(str(label).lower(), np.nan) for label in ground_truth]
    pred_integers = [label_to_int.get(str(label).lower(), np.nan) for label in predicted]
    
    # Remove any NaN values (invalid labels)
    valid_indices = [i for i in range(len(gt_integers)) 
                    if not np.isnan(gt_integers[i]) and not np.isnan(pred_integers[i])]
    
    if not valid_indices:
        return np.nan
    
    gt_valid = [gt_integers[i] for i in valid_indices]
    pred_valid = [pred_integers[i] for i in valid_indices]
    
    # Calculate RMSE
    squared_diffs = [(gt - pred) ** 2 for gt, pred in zip(gt_valid, pred_valid)]
    mse = np.mean(squared_diffs)
    rmse = np.sqrt(mse)
    
    return rmse


def calculate_krippendorff_alpha(ground_truth, predicted):
    """
    Calculate Krippendorff's alpha for ordinal data treating model predictions and ground truth as separate annotators
    
    Label mapping (ordinal scale):
    not applicable - 0
    below emerging - 1
    emerging - 2
    expanding - 3
    exemplifying - 4
    
    Args:
        ground_truth: Series or list of ground truth labels
        predicted: Series or list of predicted labels
        
    Returns:
        float: Krippendorff's alpha value
    """
    label_to_int = {
        'not applicable': 0,
        'below emerging': 1,
        'emerging': 2,
        'expanding': 3,
        'exemplifying': 4
    }
    
    # Convert labels to integers
    gt_integers = [label_to_int.get(str(label).lower(), np.nan) for label in ground_truth]
    pred_integers = [label_to_int.get(str(label).lower(), np.nan) for label in predicted]
    
    # Remove any NaN values (invalid labels)
    valid_indices = [i for i in range(len(gt_integers)) 
                    if not np.isnan(gt_integers[i]) and not np.isnan(pred_integers[i])]
    
    if len(valid_indices) < 2:  # Need at least 2 valid pairs
        return np.nan
    
    gt_valid = [gt_integers[i] for i in valid_indices]
    pred_valid = [pred_integers[i] for i in valid_indices]
    
    # Create reliability data matrix (2 annotators x n units)
    # Row 0: Ground truth annotator, Row 1: Model annotator
    reliability_data = np.array([gt_valid, pred_valid])
    
    # Calculate Krippendorff's alpha for ordinal data
    try:
        alpha = krippendorff.alpha(reliability_data, level_of_measurement='ordinal')
        return alpha if not np.isnan(alpha) else 0.0
    except Exception as e:
        print(f"Warning: Could not calculate Krippendorff's alpha: {e}")
        return 0.0


def process_subskill_metrics(results_df, result_suffix, prefix, results_dir):
    """
    Process and display metrics for each subskill
    
    Args:
        results_df (pd.DataFrame): DataFrame with ground truth and predictions
        
    Returns:
        dict: Dictionary containing metrics for each subskill
    """
    print("=" * 80)
    print("METRICS PER SUBSKILL")
    print("=" * 80)
    
    # Create directory for subskill-wise results
    subskill_results_dir = f"{results_dir}/subskill_wise_results"
    os.makedirs(subskill_results_dir, exist_ok=True)
    
    subskill_metrics = {}
    for subskill in results_df['Subskill'].unique():
        subskill_data = results_df[results_df['Subskill'] == subskill]
        
        # Calculate accuracy for this subskill
        accuracy = (subskill_data['Ground Truth'] == subskill_data['Predicted']).mean()
        
        # Calculate RMSE for this subskill
        rmse = calculate_rmse(subskill_data['Ground Truth'], subskill_data['Predicted'])
        
        # Calculate Krippendorff's alpha for this subskill
        alpha = calculate_krippendorff_alpha(subskill_data['Ground Truth'], subskill_data['Predicted'])
        
        print(f"\n{subskill}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Krippendorff's α: {alpha:.4f}")
        print(f"  Total samples: {len(subskill_data)}")
        
        # Create a safe filename for the subskill
        safe_filename = subskill.replace(':', '_').replace(' ', '_').replace('/', '_')
        output_file = os.path.join(subskill_results_dir, f"{safe_filename}.txt")
        
        # Classification report for this subskill
        try:
            # Display detailed classification report
            print(f"\n  Classification Report for {subskill}:")
            report_str = classification_report(
                subskill_data['Ground Truth'], 
                subskill_data['Predicted'], 
                zero_division=0
            )
            print("  " + report_str.replace('\n', '\n  '))
            
            # Display confusion matrix
            print(f"\n  Confusion Matrix for {subskill}:")
            cm = confusion_matrix(subskill_data['Ground Truth'], subskill_data['Predicted'])
            print("  " + str(cm).replace('\n', '\n  '))
            
            # Get metrics for storage
            report = classification_report(
                subskill_data['Ground Truth'], 
                subskill_data['Predicted'], 
                zero_division=0,
                output_dict=True
            )
            
            # Extract key metrics
            macro_avg = report.get('macro avg', {})
            weighted_avg = report.get('weighted avg', {})
            
            print(f"\n  Summary Metrics:")
            print(f"    Precision (macro): {macro_avg.get('precision', 0):.4f}")
            print(f"    Recall (macro): {macro_avg.get('recall', 0):.4f}")
            print(f"    F1-score (macro): {macro_avg.get('f1-score', 0):.4f}")
            print(f"    Precision (weighted): {weighted_avg.get('precision', 0):.4f}")
            print(f"    Recall (weighted): {weighted_avg.get('recall', 0):.4f}")
            print(f"    F1-score (weighted): {weighted_avg.get('f1-score', 0):.4f}")
            print(f"    RMSE: {rmse:.4f}")
            print(f"    Krippendorff's α: {alpha:.4f}")
            
            # Write detailed results to individual file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"RESULTS FOR {subskill}\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"RMSE: {rmse:.4f}\n")
                f.write(f"Krippendorff's α: {alpha:.4f}\n")
                f.write(f"Total samples: {len(subskill_data)}\n\n")
                
                f.write("CLASSIFICATION REPORT:\n")
                f.write("-" * 40 + "\n")
                f.write(report_str + "\n\n")
                
                f.write("CONFUSION MATRIX:\n")
                f.write("-" * 40 + "\n")
                f.write(str(cm) + "\n\n")
                
                f.write("SUMMARY METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Precision (macro): {macro_avg.get('precision', 0):.4f}\n")
                f.write(f"Recall (macro): {macro_avg.get('recall', 0):.4f}\n")
                f.write(f"F1-score (macro): {macro_avg.get('f1-score', 0):.4f}\n")
                f.write(f"Precision (weighted): {weighted_avg.get('precision', 0):.4f}\n")
                f.write(f"Recall (weighted): {weighted_avg.get('recall', 0):.4f}\n")
                f.write(f"F1-score (weighted): {weighted_avg.get('f1-score', 0):.4f}\n")
                f.write(f"RMSE: {rmse:.4f}\n")
                f.write(f"Krippendorff's α: {alpha:.4f}\n\n")
                
                f.write("DETAILED PREDICTIONS:\n")
                f.write("-" * 40 + "\n")
                for idx, row in subskill_data.iterrows():
                    f.write(f"Essay ID: {row['Essay ID']}, GT: {row['Ground Truth']}, Pred: {row['Predicted']}\n")
            
            print(f"  Results saved to: {output_file}")
            
            # Store metrics for summary
            subskill_metrics[subskill] = {
                'accuracy': accuracy,
                'rmse': rmse,
                'alpha': alpha,
                'total_samples': len(subskill_data),
                'precision_macro': macro_avg.get('precision', 0),
                'recall_macro': macro_avg.get('recall', 0),
                'f1_macro': macro_avg.get('f1-score', 0),
                'precision_weighted': weighted_avg.get('precision', 0),
                'recall_weighted': weighted_avg.get('recall', 0),
                'f1_weighted': weighted_avg.get('f1-score', 0)
            }
            
        except Exception as e:
            print(f"  Error calculating detailed metrics: {e}")
            subskill_metrics[subskill] = {
                'accuracy': accuracy,
                'rmse': rmse,
                'alpha': alpha,
                'total_samples': len(subskill_data)
            }
        
        print("-" * 60)  # Separator between subskills
    
    return subskill_metrics

def analyze_gen_subskills_results(model_type="bert", wrubric=False,
                                   traits_input=False, traits_output=False,
                                   trait_observations=False, obs_only=False):
    """
    Analyze results from subskill-based fine-tuning experiments
    UNIQUE_GEN_SUBSKILLS_FUNCTION
    """
    model_name = model_type.upper()
    if wrubric:
        model_name += "_WRUBRIC"

    traits_suffix = _traits_dir_suffix(traits_input, traits_output, trait_observations, obs_only)
    if traits_suffix:
        model_name += traits_suffix.upper()

    print("=" * 80)
    print(f"ANALYZING {model_name} SUBSKILL-BASED FINE-TUNING EXPERIMENT RESULTS")
    print("=" * 80)
    
    if model_type == "llama_no_just":
        base_dir = "llama_results_no_just_wrubric_gen_subskills" if wrubric else "llama_results_no_just_gen_subskills"
    elif model_type == "llama":
        base_dir = "llama_results_wrubric_gen_subskills" if wrubric else "llama_results_gen_subskills"
    elif model_type == "modernbert":
        base_dir = "modernbert_results_wrubric_gen_subskills" if wrubric else "modernbert_results_gen_subskills"
    else:
        base_dir = f"{model_type}_results_gen_subskills"

    base_dir = base_dir + traits_suffix

    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} not found!")
        print(f"Please run: python {model_type}_classify.py --gen_subskills first")
        return
    
    # Load ground truth data
    gt_df = pd.read_excel("combined_essays.xlsx")
    gt_df.columns = gt_df.iloc[0]
    gt_df = gt_df[1:]
    
    # Define the experiments and their target subskills
    experiments = [
        {
            'name': 'subskill_4',
            'test_subskills': ['4.1', '4.2'],
            'train_subskills': ['2.1', '2.2', '3.1', '3.2'],
            'description': 'Train on 2,3 -> Test on 4 (subskill-based split)'
        },
        {
            'name': 'subskill_3', 
            'test_subskills': ['3.1', '3.2'],
            'train_subskills': ['2.1', '2.2', '4.1', '4.2'],
            'description': 'Train on 2,4 -> Test on 3 (subskill-based split)'
        },
        {
            'name': 'subskill_2',
            'test_subskills': ['2.1', '2.2'],
            'train_subskills': ['3.1', '3.2', '4.1', '4.2'], 
            'description': 'Train on 3,4 -> Test on 2 (subskill-based split)'
        }
    ]
    
    # Find experiment directories
    exp_dirs = []
    for exp in experiments:
        exp_dir = os.path.join(base_dir, exp['name'])
        if os.path.exists(exp_dir):
            exp_dirs.append((exp, exp_dir))
        else:
            print(f"Warning: Experiment directory {exp_dir} not found!")
    
    if not exp_dirs:
        print(f"Error: No experiment directories found in {base_dir}")
        return
    
    print(f"Found {len(exp_dirs)} experiment directories")
    
    # Create overall results directory
    overall_results_dir = os.path.join(base_dir, "overall_results")
    os.makedirs(overall_results_dir, exist_ok=True)
    
    all_results = {}
    
    # Collect all predictions across all experiments for overall metrics
    all_predictions = []
    all_ground_truth = []
    
    # Subskill map for all functions
    subskill_map = {
        '2.1': '2.1: Synthesizing Multiple Sources',
        '2.2': '2.2: Evaluating Evidence Strength',
        '3.1': '3.1: Using Counterarguments',
        '3.2': '3.2: Using Facts and Opinions',
        '4.1': '4.1: Drawing Conclusions',
        '4.2': '4.2: Using Logical Fallacies'
    }
    
    # Process each experiment
    for exp_config, exp_dir in exp_dirs:
        print(f"\n{'-'*60}")
        print(f"Processing: {exp_config['description']}")
        print(f"Directory: {exp_dir}")
        print(f"Test subskills: {exp_config['test_subskills']}")
        print(f"Train subskills: {exp_config['train_subskills']}")
        
        # Load test results
        test_results_file = os.path.join(exp_dir, "test_results.json")
        if not os.path.exists(test_results_file):
            print(f"Error: {test_results_file} not found!")
            continue
        
        with open(test_results_file, "r", encoding="utf-8") as f:
            pred_results = json.load(f)
        
        # Build results dataframe ONLY for test subskills
        results_data = []
        
        for _, row in gt_df.iterrows():
            essay_id = row['essay_id']
            if essay_id not in pred_results:
                continue
            pred_rubric = pred_results[essay_id]
            for subskill, gt_label in row.items():
                if 'Subskill' not in subskill:
                    continue
                subskill_number = subskill.split('Subskill ')[-1]
                subskill_name = subskill_map.get(subskill_number, subskill)
                pred_label = pred_rubric.get(subskill_name, {}).get('label', None)
                if pred_label not in ['Few-shot Example', 'Parsing Error'] and pred_label is not None:
                    # Only include test subskills for this experiment
                    if subskill_number in exp_config['test_subskills']:
                        results_data.append({
                            'Essay ID': essay_id,
                            'Subskill': subskill,
                            'Ground Truth': gt_label.lower().strip(),
                            'Predicted': pred_label.lower().strip()
                        })
                        # Collect for overall metrics across all experiments
                        all_ground_truth.append(gt_label.lower().strip())
                        all_predictions.append(pred_label.lower().strip())
        
        # Calculate overall metrics for this experiment (combined test subskills only)
        if results_data:
            experiment_df = pd.DataFrame(results_data)
            from sklearn.metrics import classification_report
            
            # Calculate combined metrics on test subskills only
            overall_accuracy = (experiment_df['Ground Truth'] == experiment_df['Predicted']).mean()
            overall_rmse = calculate_rmse(experiment_df['Ground Truth'], experiment_df['Predicted'])
            overall_alpha = calculate_krippendorff_alpha(experiment_df['Ground Truth'], experiment_df['Predicted'])
            
            # Get F1 scores from classification report
            exp_report_dict = classification_report(experiment_df['Ground Truth'], experiment_df['Predicted'], 
                                                   zero_division=0, output_dict=True)
            exp_macro = exp_report_dict.get('macro avg', {})
            exp_weighted = exp_report_dict.get('weighted avg', {})
            
            overall_f1_macro = exp_macro.get('f1-score', 0)
            overall_f1_weighted = exp_weighted.get('f1-score', 0)
            total_samples = len(experiment_df)
            
            # Process subskill-wise results (for individual subskill metrics)
            subskill_metrics = process_subskill_metrics(experiment_df, "", "", exp_dir)
        else:
            overall_f1_macro = 0
            overall_f1_weighted = 0
            total_samples = 0
        
        # Filter subskill metrics to only include test subskills
        filtered_subskill_metrics = {}
        for subskill_key, metrics in subskill_metrics.items():
            subskill_num = subskill_key.replace("Subskill ", "")
            if subskill_num in exp_config['test_subskills']:
                filtered_subskill_metrics[subskill_key] = metrics
        
        # Store results for overall summary
        all_results[exp_config['name']] = {
            'description': exp_config['description'],
            'test_subskills': exp_config['test_subskills'],
            'train_subskills': exp_config['train_subskills'],
            'overall_accuracy': overall_accuracy,
            'overall_rmse': overall_rmse,
            'overall_alpha': overall_alpha,
            'overall_f1_macro': overall_f1_macro,
            'overall_f1_weighted': overall_f1_weighted,
            'total_samples': total_samples,
            'subskill_metrics': filtered_subskill_metrics
        }
        
        print(f"✅ Results processed for {exp_config['name']}")
        print(f"   Overall Accuracy: {overall_accuracy:.4f}")
        print(f"   Overall RMSE: {overall_rmse:.4f}")
        print(f"   Overall Krippendorff's α: {overall_alpha:.4f}")
        print(f"   Overall F1-macro: {overall_f1_macro:.4f}")
        print(f"   Overall F1-weighted: {overall_f1_weighted:.4f}")
        print(f"   Total samples: {total_samples}")
        print(f"   Test subskills: {list(filtered_subskill_metrics.keys())}")
    
    # Create overall summary
    print(f"\n{'='*80}")
    print(f"{model_name} SUBSKILL-BASED FINE-TUNING EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    # Save overall summary file
    summary_file = os.path.join(overall_results_dir, "gen_subskills_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"{model_name} SUBSKILL-BASED FINE-TUNING EXPERIMENT SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write("Methodology: Split by subskills (not essays)\n")
        f.write("- Training subskills: 90% train, 10% validation (across all essays)\n")
        f.write("- Test subskill: 100% of essays for testing\n\n")
        
        for exp_name, results in all_results.items():
            print(f"\n{results['description']}:")
            print(f"  Training subskills: {', '.join(results['train_subskills'])}")
            print(f"  Test subskills: {', '.join(results['test_subskills'])}")
            print(f"  Overall Accuracy: {results['overall_accuracy']:.4f}")
            print(f"  Overall RMSE: {results['overall_rmse']:.4f}")
            print(f"  Overall Krippendorff's α: {results['overall_alpha']:.4f}")
            print(f"  Overall F1-macro: {results['overall_f1_macro']:.4f}")
            print(f"  Overall F1-weighted: {results['overall_f1_weighted']:.4f}")
            print(f"  Total samples: {results['total_samples']}")
            
            f.write(f"{results['description']}:\n")
            f.write(f"  Training subskills: {', '.join(results['train_subskills'])}\n")
            f.write(f"  Test subskills: {', '.join(results['test_subskills'])}\n")
            f.write(f"  Overall Accuracy: {results['overall_accuracy']:.4f}\n")
            f.write(f"  Overall RMSE: {results['overall_rmse']:.4f}\n")
            f.write(f"  Overall Krippendorff's α: {results['overall_alpha']:.4f}\n")
            f.write(f"  Overall F1-macro: {results['overall_f1_macro']:.4f}\n")
            f.write(f"  Overall F1-weighted: {results['overall_f1_weighted']:.4f}\n")
            f.write(f"  Total samples: {results['total_samples']}\n")
            
            # Show individual subskill results for test subskills only
            for subskill, metrics in results['subskill_metrics'].items():
                subskill_clean = subskill.replace("Subskill ", "")
                accuracy = metrics.get('accuracy', 0)
                rmse = metrics.get('rmse', 0)
                alpha = metrics.get('alpha', 0)
                total_samples = metrics.get('total_samples', 0)
                f1_macro = metrics.get('f1_macro', 0)
                f1_weighted = metrics.get('f1_weighted', 0)
                
                print(f"    {subskill_clean}: Acc={accuracy:.4f}, RMSE={rmse:.4f}, α={alpha:.4f}, F1-macro={f1_macro:.4f}, F1-weighted={f1_weighted:.4f}, N={total_samples}")
                f.write(f"    {subskill_clean}: Acc={accuracy:.4f}, RMSE={rmse:.4f}, α={alpha:.4f}, F1-macro={f1_macro:.4f}, F1-weighted={f1_weighted:.4f}, N={total_samples}\n")
            
            f.write("\n")
    
    # Compute overall aggregated metrics across ALL experiments (entire dataset)
    if all_ground_truth and all_predictions:
        from sklearn.metrics import classification_report
        
        overall_combined_accuracy = sum([1 for gt, pred in zip(all_ground_truth, all_predictions) if gt == pred]) / len(all_ground_truth)
        overall_combined_rmse = calculate_rmse(all_ground_truth, all_predictions)
        overall_combined_alpha = calculate_krippendorff_alpha(all_ground_truth, all_predictions)
        
        # Get F1 scores from classification report
        combined_report_dict = classification_report(all_ground_truth, all_predictions, 
                                                     zero_division=0, output_dict=True)
        combined_macro = combined_report_dict.get('macro avg', {})
        combined_weighted = combined_report_dict.get('weighted avg', {})
        
        overall_combined_f1_macro = combined_macro.get('f1-score', 0)
        overall_combined_f1_weighted = combined_weighted.get('f1-score', 0)
        total_combined_samples = len(all_ground_truth)
        
        # Print overall aggregated results
        print(f"\n{'='*80}")
        print(f"OVERALL AGGREGATED RESULTS (ALL SUBSKILLS)")
        print(f"{'='*80}")
        print(f"  Overall Accuracy: {overall_combined_accuracy:.4f}")
        print(f"  Overall RMSE: {overall_combined_rmse:.4f}")
        print(f"  Overall Krippendorff's α: {overall_combined_alpha:.4f}")
        print(f"  Overall F1-macro: {overall_combined_f1_macro:.4f}")
        print(f"  Overall F1-weighted: {overall_combined_f1_weighted:.4f}")
        print(f"  Total samples: {total_combined_samples}")
        
        # Append to summary file
        with open(summary_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"OVERALL AGGREGATED RESULTS (ALL SUBSKILLS)\n")
            f.write(f"{'='*80}\n")
            f.write(f"  Overall Accuracy: {overall_combined_accuracy:.4f}\n")
            f.write(f"  Overall RMSE: {overall_combined_rmse:.4f}\n")
            f.write(f"  Overall Krippendorff's α: {overall_combined_alpha:.4f}\n")
            f.write(f"  Overall F1-macro: {overall_combined_f1_macro:.4f}\n")
            f.write(f"  Overall F1-weighted: {overall_combined_f1_weighted:.4f}\n")
            f.write(f"  Total samples: {total_combined_samples}\n")
            f.write(f"  Note: Aggregated predictions from all experiments (subskill_2, subskill_3, subskill_4)\n")
    else:
        # Initialize with default values if no data
        overall_combined_accuracy = 0
        overall_combined_rmse = 0
        overall_combined_alpha = 0
        overall_combined_f1_macro = 0
        overall_combined_f1_weighted = 0
        total_combined_samples = 0
    
    # Create CSV summary for easy analysis
    summary_data = []
    
    # Add overall aggregated row first
    if all_ground_truth and all_predictions:
        summary_data.append({
            'Experiment': 'overall',
            'Description': 'Overall across all experiments (entire dataset)',
            'Train_Subskills': '2.1, 2.2, 3.1, 3.2, 4.1, 4.2',
            'Test_Subskills': '2.1, 2.2, 3.1, 3.2, 4.1, 4.2',
            'Split_Method': 'subskill_based',
            'Overall_Accuracy': round(overall_combined_accuracy, 4),
            'Overall_RMSE': round(overall_combined_rmse, 4),
            'Overall_Alpha': round(overall_combined_alpha, 4),
            'F1_Macro': round(overall_combined_f1_macro, 4),
            'F1_Weighted': round(overall_combined_f1_weighted, 4),
            'Total_Samples': total_combined_samples
        })
    
    for exp_name, results in all_results.items():
        summary_data.append({
            'Experiment': exp_name,
            'Description': results['description'],
            'Train_Subskills': ', '.join(results['train_subskills']),
            'Test_Subskills': ', '.join(results['test_subskills']),
            'Split_Method': 'subskill_based',
            'Overall_Accuracy': round(results['overall_accuracy'], 4),
            'Overall_RMSE': round(results['overall_rmse'], 4),
            'Overall_Alpha': round(results['overall_alpha'], 4),
            'F1_Macro': round(results['overall_f1_macro'], 4),
            'F1_Weighted': round(results['overall_f1_weighted'], 4),
            'Total_Samples': results['total_samples']
        })
        
        # Add individual subskill rows
        for subskill, metrics in results['subskill_metrics'].items():
            subskill_clean = subskill.replace("Subskill ", "")
            summary_data.append({
                'Experiment': f"{exp_name}_{subskill_clean}",
                'Description': f"{results['description']} - {subskill_clean}",
                'Train_Subskills': ', '.join(results['train_subskills']),
                'Test_Subskills': subskill_clean,
                'Split_Method': 'subskill_based',
                'Overall_Accuracy': round(metrics.get('accuracy', 0), 4),
                'Overall_RMSE': round(metrics.get('rmse', 0), 4),
                'Overall_Alpha': round(metrics.get('alpha', 0), 4),
                'F1_Macro': round(metrics.get('f1_macro', 0), 4),
                'F1_Weighted': round(metrics.get('f1_weighted', 0), 4),
                'Total_Samples': metrics.get('total_samples', 0)
            })
    
    summary_df = pd.DataFrame(summary_data)
    # Round all float columns to 4 decimal places before saving
    float_columns = summary_df.select_dtypes(include=[np.float64, np.float32]).columns
    summary_df[float_columns] = summary_df[float_columns].round(4)
    summary_csv = os.path.join(overall_results_dir, "gen_subskills_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    
    print(f"\nSubskill-based fine-tuning results saved to: {base_dir}/overall_results/")
    print(f"  - gen_subskills_summary.txt: Detailed summary")
    print(f"  - gen_subskills_summary.csv: Machine-readable summary")
    
    print(f"\nIndividual experiment results available in:")
    for exp_config, exp_dir in exp_dirs:
        print(f"  - {exp_dir}/ ({exp_config['description']})")

def _process_gen_subskills_subset(exp_dirs, gt_df, subset_ids):
    """
    Internal helper: compute gen_subskills-style metrics restricted to a specific
    set of essay IDs (e.g. the 80% train+val or 20% test essays from the
    --generalize random_state=47 split).

    Returns: (all_results dict, combined_gt list, combined_pred list)
    """
    subskill_map = {
        '2.1': '2.1: Synthesizing Multiple Sources',
        '2.2': '2.2: Evaluating Evidence Strength',
        '3.1': '3.1: Using Counterarguments',
        '3.2': '3.2: Using Facts and Opinions',
        '4.1': '4.1: Drawing Conclusions',
        '4.2': '4.2: Using Logical Fallacies'
    }
    all_results = {}
    combined_gt = []
    combined_pred = []

    for exp_config, exp_dir in exp_dirs:
        test_results_file = os.path.join(exp_dir, "test_results.json")
        if not os.path.exists(test_results_file):
            print(f"  Warning: {test_results_file} not found, skipping.")
            continue
        with open(test_results_file, "r", encoding="utf-8") as f:
            pred_results = json.load(f)

        results_data = []
        for _, row in gt_df.iterrows():
            essay_id = row['essay_id']
            if essay_id not in subset_ids:
                continue
            if essay_id not in pred_results:
                continue
            pred_rubric = pred_results[essay_id]
            for subskill, gt_label in row.items():
                if 'Subskill' not in subskill:
                    continue
                subskill_number = subskill.split('Subskill ')[-1]
                subskill_name = subskill_map.get(subskill_number, subskill)
                pred_label = pred_rubric.get(subskill_name, {}).get('label', None)
                if pred_label not in ['Few-shot Example', 'Parsing Error'] and pred_label is not None:
                    if subskill_number in exp_config['test_subskills']:
                        results_data.append({
                            'Essay ID': essay_id,
                            'Subskill': subskill,
                            'Ground Truth': gt_label.lower().strip(),
                            'Predicted': pred_label.lower().strip()
                        })
                        combined_gt.append(gt_label.lower().strip())
                        combined_pred.append(pred_label.lower().strip())

        if results_data:
            experiment_df = pd.DataFrame(results_data)
            from sklearn.metrics import classification_report as _cr
            overall_accuracy = (experiment_df['Ground Truth'] == experiment_df['Predicted']).mean()
            overall_rmse = calculate_rmse(experiment_df['Ground Truth'], experiment_df['Predicted'])
            overall_alpha = calculate_krippendorff_alpha(experiment_df['Ground Truth'], experiment_df['Predicted'])
            rep = _cr(experiment_df['Ground Truth'], experiment_df['Predicted'], zero_division=0, output_dict=True)
            overall_f1_macro = rep.get('macro avg', {}).get('f1-score', 0)
            overall_f1_weighted = rep.get('weighted avg', {}).get('f1-score', 0)
            total_samples = len(experiment_df)
            subskill_metrics = process_subskill_metrics(experiment_df, "", "", exp_dir)
        else:
            overall_accuracy = overall_rmse = overall_alpha = 0
            overall_f1_macro = overall_f1_weighted = 0
            total_samples = 0
            subskill_metrics = {}

        filtered_subskill_metrics = {
            k: v for k, v in subskill_metrics.items()
            if k.replace("Subskill ", "") in exp_config['test_subskills']
        }

        all_results[exp_config['name']] = {
            'description': exp_config['description'],
            'test_subskills': exp_config['test_subskills'],
            'train_subskills': exp_config['train_subskills'],
            'overall_accuracy': overall_accuracy,
            'overall_rmse': overall_rmse,
            'overall_alpha': overall_alpha,
            'overall_f1_macro': overall_f1_macro,
            'overall_f1_weighted': overall_f1_weighted,
            'total_samples': total_samples,
            'subskill_metrics': filtered_subskill_metrics
        }

    return all_results, combined_gt, combined_pred


def _build_gen_subskills_csv(all_results, combined_gt, combined_pred):
    """
    Build a summary_data list (for pd.DataFrame) from gen_subskills experiment
    results and combined overall metrics, matching gen_subskills_summary.csv format.
    """
    summary_data = []

    if combined_gt and combined_pred:
        from sklearn.metrics import classification_report as _cr
        overall_combined_accuracy = sum(g == p for g, p in zip(combined_gt, combined_pred)) / len(combined_gt)
        overall_combined_rmse = calculate_rmse(combined_gt, combined_pred)
        overall_combined_alpha = calculate_krippendorff_alpha(combined_gt, combined_pred)
        rep = _cr(combined_gt, combined_pred, zero_division=0, output_dict=True)
        overall_combined_f1_macro = rep.get('macro avg', {}).get('f1-score', 0)
        overall_combined_f1_weighted = rep.get('weighted avg', {}).get('f1-score', 0)
        total_combined_samples = len(combined_gt)

        summary_data.append({
            'Experiment': 'overall',
            'Description': 'Overall across all experiments',
            'Train_Subskills': '2.1, 2.2, 3.1, 3.2, 4.1, 4.2',
            'Test_Subskills': '2.1, 2.2, 3.1, 3.2, 4.1, 4.2',
            'Split_Method': 'subskill_based',
            'Overall_Accuracy': round(overall_combined_accuracy, 4),
            'Overall_RMSE': round(overall_combined_rmse, 4),
            'Overall_Alpha': round(overall_combined_alpha, 4),
            'F1_Macro': round(overall_combined_f1_macro, 4),
            'F1_Weighted': round(overall_combined_f1_weighted, 4),
            'Total_Samples': total_combined_samples
        })

    for exp_name, results in all_results.items():
        summary_data.append({
            'Experiment': exp_name,
            'Description': results['description'],
            'Train_Subskills': ', '.join(results['train_subskills']),
            'Test_Subskills': ', '.join(results['test_subskills']),
            'Split_Method': 'subskill_based',
            'Overall_Accuracy': round(results['overall_accuracy'], 4),
            'Overall_RMSE': round(results['overall_rmse'], 4),
            'Overall_Alpha': round(results['overall_alpha'], 4),
            'F1_Macro': round(results['overall_f1_macro'], 4),
            'F1_Weighted': round(results['overall_f1_weighted'], 4),
            'Total_Samples': results['total_samples']
        })
        for subskill, metrics in results['subskill_metrics'].items():
            subskill_clean = subskill.replace("Subskill ", "")
            summary_data.append({
                'Experiment': f"{exp_name}_{subskill_clean}",
                'Description': f"{results['description']} - {subskill_clean}",
                'Train_Subskills': ', '.join(results['train_subskills']),
                'Test_Subskills': subskill_clean,
                'Split_Method': 'subskill_based',
                'Overall_Accuracy': round(metrics.get('accuracy', 0), 4),
                'Overall_RMSE': round(metrics.get('rmse', 0), 4),
                'Overall_Alpha': round(metrics.get('alpha', 0), 4),
                'F1_Macro': round(metrics.get('f1_macro', 0), 4),
                'F1_Weighted': round(metrics.get('f1_weighted', 0), 4),
                'Total_Samples': metrics.get('total_samples', 0)
            })

    return summary_data


def analyze_gen_subskills_split_results(model_type="llama", wrubric=False,
                                        traits_input=False, traits_output=False,
                                        trait_observations=False, obs_only=False):
    """
    Like analyze_gen_subskills_results() but reports metrics separately for:
      - train+val essays (80%): the 70%+10% subset from the random_state=47 split
      - test essays (20%):     the held-out 20% subset

    The --gen_subskills test_results.json contains predictions for ALL essays
    (since the subskill split covers all 500). This function recovers the
    --generalize essay split (random_state=47) to partition those predictions.

    Outputs:
      {base_dir}/overall_results_essay_split/train_val_summary.csv
      {base_dir}/overall_results_essay_split/test_summary.csv
    UNIQUE_GEN_SUBSKILLS_SPLIT_FUNCTION
    """
    model_name = model_type.upper()
    if wrubric:
        model_name += "_WRUBRIC"

    traits_suffix = _traits_dir_suffix(traits_input, traits_output, trait_observations, obs_only)
    if traits_suffix:
        model_name += traits_suffix.upper()

    print("=" * 80)
    print(f"ANALYZING {model_name} GEN_SUBSKILLS RESULTS — ESSAY SPLIT (train+val vs test)")
    print("=" * 80)

    if model_type == "llama_no_just":
        base_dir = "llama_results_no_just_wrubric_gen_subskills" if wrubric else "llama_results_no_just_gen_subskills"
    elif model_type == "llama":
        base_dir = "llama_results_wrubric_gen_subskills" if wrubric else "llama_results_gen_subskills"
    elif model_type == "modernbert":
        base_dir = "modernbert_results_wrubric_gen_subskills" if wrubric else "modernbert_results_gen_subskills"
    else:
        base_dir = f"{model_type}_results_gen_subskills"

    base_dir = base_dir + traits_suffix

    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} not found!")
        print(f"Please run: python {model_type}_classify.py --gen_subskills first")
        return

    # Load ground truth data
    gt_df = pd.read_excel("combined_essays.xlsx")
    gt_df.columns = gt_df.iloc[0]
    gt_df = gt_df[1:]

    # Recover the same 70/10/20 essay split used by --generalize
    train_df, val_df, test_df = create_data_splits(gt_df, random_state=47)
    train_val_ids = set(train_df['essay_id'].tolist()) | set(val_df['essay_id'].tolist())
    test_ids = set(test_df['essay_id'].tolist())

    print(f"Essay split (random_state=47): {len(train_val_ids)} train+val, {len(test_ids)} test")

    # Define experiments
    experiments = [
        {
            'name': 'subskill_4',
            'test_subskills': ['4.1', '4.2'],
            'train_subskills': ['2.1', '2.2', '3.1', '3.2'],
            'description': 'Train on 2,3 -> Test on 4 (subskill-based split)'
        },
        {
            'name': 'subskill_3',
            'test_subskills': ['3.1', '3.2'],
            'train_subskills': ['2.1', '2.2', '4.1', '4.2'],
            'description': 'Train on 2,4 -> Test on 3 (subskill-based split)'
        },
        {
            'name': 'subskill_2',
            'test_subskills': ['2.1', '2.2'],
            'train_subskills': ['3.1', '3.2', '4.1', '4.2'],
            'description': 'Train on 3,4 -> Test on 2 (subskill-based split)'
        }
    ]

    exp_dirs = []
    for exp in experiments:
        exp_dir = os.path.join(base_dir, exp['name'])
        if os.path.exists(exp_dir):
            exp_dirs.append((exp, exp_dir))
        else:
            print(f"Warning: Experiment directory {exp_dir} not found!")

    if not exp_dirs:
        print(f"Error: No experiment directories found in {base_dir}")
        return

    print(f"Found {len(exp_dirs)} experiment directories")

    overall_results_dir = os.path.join(base_dir, "overall_results_essay_split")
    os.makedirs(overall_results_dir, exist_ok=True)

    # --- Process train+val subset ---
    print(f"\n{'='*60}\nProcessing TRAIN+VAL subset ({len(train_val_ids)} essays)")
    tv_results, tv_gt, tv_pred = _process_gen_subskills_subset(exp_dirs, gt_df, train_val_ids)

    print(f"\n{'='*60}\nProcessing TEST subset ({len(test_ids)} essays)")
    te_results, te_gt, te_pred = _process_gen_subskills_subset(exp_dirs, gt_df, test_ids)

    # --- Print combined summaries ---
    for label, results, combined_gt, combined_pred in [
        ("TRAIN+VAL", tv_results, tv_gt, tv_pred),
        ("TEST", te_results, te_gt, te_pred),
    ]:
        print(f"\n{'='*80}")
        print(f"{model_name} GEN_SUBSKILLS — {label} SUBSET SUMMARY")
        print(f"{'='*80}")
        for exp_name, res in results.items():
            print(f"  {res['description']}: Acc={res['overall_accuracy']:.4f}, "
                  f"RMSE={res['overall_rmse']:.4f}, α={res['overall_alpha']:.4f}, "
                  f"F1-mac={res['overall_f1_macro']:.4f}, N={res['total_samples']}")
        if combined_gt:
            from sklearn.metrics import classification_report as _cr
            acc = sum(g == p for g, p in zip(combined_gt, combined_pred)) / len(combined_gt)
            alpha = calculate_krippendorff_alpha(combined_gt, combined_pred)
            rep = _cr(combined_gt, combined_pred, zero_division=0, output_dict=True)
            print(f"  OVERALL ({label}): Acc={acc:.4f}, α={alpha:.4f}, "
                  f"F1-mac={rep.get('macro avg',{}).get('f1-score',0):.4f}, N={len(combined_gt)}")

    # --- Save CSVs ---
    for label, fname, results, combined_gt, combined_pred in [
        ("train+val", "train_val_summary.csv", tv_results, tv_gt, tv_pred),
        ("test",      "test_summary.csv",      te_results, te_gt, te_pred),
    ]:
        summary_data = _build_gen_subskills_csv(results, combined_gt, combined_pred)
        summary_df = pd.DataFrame(summary_data)
        float_cols = summary_df.select_dtypes(include=[np.float64, np.float32]).columns
        summary_df[float_cols] = summary_df[float_cols].round(4)
        out_path = os.path.join(overall_results_dir, fname)
        summary_df.to_csv(out_path, index=False)
        print(f"\nSaved {label} summary → {out_path}")

    print(f"\nEssay-split results saved to: {base_dir}/overall_results_essay_split/")
    print(f"  - train_val_summary.csv : metrics on 80%% train+val essays")
    print(f"  - test_summary.csv      : metrics on 20%% test essays")


def parse_args():
    parser = argparse.ArgumentParser(description="Classification of student essays")
    parser.add_argument("--bert", action="store_true", help="Consider BERT model results")
    parser.add_argument("--longformer", action="store_true", help="Consider Longformer model results")
    parser.add_argument("--modernbert", action="store_true", help="Consider ModernBERT model results")
    parser.add_argument("--llama", action="store_true", help="Consider LLaMA model results")
    parser.add_argument("--llama_no_just", action="store_true", help="Consider LLaMA No Justification model results")
    parser.add_argument("--ft_mode", action="store_true", help="Consider test set from fine-tune setup")
    parser.add_argument("--few_shot", action="store_true", help="Consider few shot prompting")
    parser.add_argument("--model_choice", type=int, default=0, help="Model Choice: 0 for gpt-4.1, 1 for gpt-5, 2 for gpt-5-mini")
    parser.add_argument("--k", type=int, default=1, help="Number of few-shot examples per category (default: 1)")
    parser.add_argument("--wrubric", action="store_true", help="Consider results from model trained with rubric info")
    parser.add_argument("--split", action="store_true", help="Analyze results from multiple random splits")
    parser.add_argument("--generalize", action="store_true", help="Analyze results from generalization experiments")
    parser.add_argument("--gen_subskills", action="store_true", help="Analyze results from subskill-based fine-tuning experiments")
    parser.add_argument("--essay_split", action="store_true", help="Used with --gen_subskills: report metrics separately for the 80%% train+val and 20%% test essay subsets (split recovered from --generalize random_state=47)")
    parser.add_argument("--gen_train", action="store_true", help="Analyze gen_train results (test_results_train.json): inference on 80%% train+val essays with --generalize checkpoints")
    parser.add_argument("--generalize_seen", action="store_true", help="Analyze --generalize_seen results: both test_results.json (20%% test) and test_results_train.json (80%% train+val) from *_generalize_seen/ dirs")
    parser.add_argument("--traits_input", action="store_true", help="Results were produced with trait labels injected into the prompt input")
    parser.add_argument("--traits_output", action="store_true", help="Results were produced with the model predicting trait labels as output")
    parser.add_argument("--trait_observations", action="store_true", help="Results were produced with per-trait observations (requires --traits_input and/or --traits_output)")
    parser.add_argument("--obs_only", action="store_true", help="Results were produced with only per-trait observations and no trait labels (requires --traits_input and/or --traits_output)")
    parser.add_argument("--contrastive", action="store_true", help="Results were produced with contrastive loss (used with --generalize)")
    return parser.parse_args()

def analyze_results():
    # parse arguments
    args = parse_args()

    # Handle split analysis first
    if args.split:
        if args.ft_mode:
            # Combined ft_mode and split analysis for LLM results
            few_shot = args.few_shot
            model_choice = args.model_choice
            k = args.k
            analyze_ft_split_results(model_choice, few_shot, k)
        elif args.bert:
            analyze_split_results("bert")
        elif args.modernbert:
            analyze_split_results("modernbert", args.wrubric)
        elif args.longformer:
            analyze_split_results("longformer")
        elif args.llama:
            analyze_split_results("llama", args.wrubric)
        elif args.llama_no_just:
            analyze_split_results("llama_no_just", args.wrubric)
        else:
            # Combined ft_mode and split analysis for LLM results
            few_shot = args.few_shot
            model_choice = args.model_choice
            k = args.k
            analyze_ft_split_results(model_choice, few_shot, k)
        return
    
    # Handle generalization analysis
    if args.generalize:
        ti = args.traits_input
        to_ = args.traits_output
        tobs = args.trait_observations
        obs_only = args.obs_only
        con = args.contrastive
        if args.bert:
            analyze_generalization_results("bert")
        elif args.modernbert:
            analyze_generalization_results("modernbert", args.wrubric)
        elif args.longformer:
            analyze_generalization_results("longformer")
        elif args.llama:
            analyze_generalization_results("llama", args.wrubric,
                                           traits_input=ti, traits_output=to_,
                                           trait_observations=tobs, obs_only=obs_only,
                                           contrastive=con)
        elif args.llama_no_just:
            analyze_generalization_results("llama_no_just", args.wrubric,
                                           traits_input=ti, traits_output=to_,
                                           trait_observations=tobs, obs_only=obs_only,
                                           contrastive=con)
        else:
            # Default to BERT for backward compatibility
            analyze_generalization_results("bert")
        return

    # Handle subskill-based fine-tuning analysis
    if args.gen_subskills:
        ti = args.traits_input
        to_ = args.traits_output
        tobs = args.trait_observations
        obs_only = args.obs_only
        fn = analyze_gen_subskills_split_results if args.essay_split else analyze_gen_subskills_results
        if args.bert:
            fn("bert")
        elif args.modernbert:
            fn("modernbert", args.wrubric)
        elif args.longformer:
            fn("longformer")
        elif args.llama:
            fn("llama", args.wrubric,
               traits_input=ti, traits_output=to_,
               trait_observations=tobs, obs_only=obs_only)
        elif args.llama_no_just:
            fn("llama_no_just", args.wrubric,
               traits_input=ti, traits_output=to_,
               trait_observations=tobs, obs_only=obs_only)
        else:
            fn("llama", args.wrubric)
        return

    # Handle gen_train analysis (inference on 80% train+val essays, held-out subskill)
    if args.gen_train:
        ti = args.traits_input
        to_ = args.traits_output
        tobs = args.trait_observations
        obs_only = args.obs_only
        con = args.contrastive
        if args.bert:
            analyze_gen_train_results("bert")
        elif args.modernbert:
            analyze_gen_train_results("modernbert", args.wrubric)
        elif args.longformer:
            analyze_gen_train_results("longformer")
        elif args.llama:
            analyze_gen_train_results("llama", args.wrubric,
                                      traits_input=ti, traits_output=to_,
                                      trait_observations=tobs, obs_only=obs_only,
                                      contrastive=con)
        elif args.llama_no_just:
            analyze_gen_train_results("llama_no_just", args.wrubric,
                                      traits_input=ti, traits_output=to_,
                                      trait_observations=tobs, obs_only=obs_only,
                                      contrastive=con)
        else:
            analyze_gen_train_results("llama", args.wrubric)
        return

    # Handle generalize_seen analysis (test_results.json + test_results_train.json from *_generalize_seen/)
    if args.generalize_seen:
        ti = args.traits_input
        to_ = args.traits_output
        tobs = args.trait_observations
        obs_only = args.obs_only
        con = args.contrastive
        if args.bert:
            analyze_generalize_seen_results("bert")
        elif args.modernbert:
            analyze_generalize_seen_results("modernbert", args.wrubric)
        elif args.longformer:
            analyze_generalize_seen_results("longformer")
        elif args.llama:
            analyze_generalize_seen_results("llama", args.wrubric,
                                            traits_input=ti, traits_output=to_,
                                            trait_observations=tobs, obs_only=obs_only,
                                            contrastive=con)
        elif args.llama_no_just:
            analyze_generalize_seen_results("llama_no_just", args.wrubric,
                                            traits_input=ti, traits_output=to_,
                                            trait_observations=tobs, obs_only=obs_only,
                                            contrastive=con)
        else:
            analyze_generalize_seen_results("llama", args.wrubric)
        return

    few_shot = args.few_shot
    k = args.k
    if few_shot:
        result_suffix = "few_shot" if k == 1 else f"few_shot_k{k}"
    else:
        result_suffix = "zero_shot"
    
    bert = args.bert
    longformer = args.longformer
    modernbert = args.modernbert
    llama = args.llama
    llama_no_just = args.llama_no_just
    ft_mode = args.ft_mode
    
    model_choice = args.model_choice
    if model_choice == 0:
        prefix = "gpt-4.1"
    elif model_choice == 1:
        prefix = "gpt-5"
    elif model_choice == 2:
        prefix = "gpt-5-mini"

    # load the ground truth data 
    gt_df = pd.read_excel("combined_essays.xlsx")
    gt_df.columns = gt_df.iloc[0]
    gt_df = gt_df[1:]

    # load the predicted results
    if bert:
        results_dir = "bert_results"
        with open(f"{results_dir}/test_results.json", "r", encoding="utf-8") as f:
            pred_results = json.load(f)
    elif longformer:
        results_dir = "longformer_results"
        with open(f"{results_dir}/test_results.json", "r", encoding="utf-8") as f:
            pred_results = json.load(f)
    elif modernbert:
        results_dir = "modernbert_results_wrubric" if args.wrubric else "modernbert_results"
        with open(f"{results_dir}/test_results.json", "r", encoding="utf-8") as f:
            pred_results = json.load(f)
    elif llama:
        results_dir = "llama_results_wrubric" if args.wrubric else "llama_results"
        with open(f"{results_dir}/test_results.json", "r", encoding="utf-8") as f:
            pred_results = json.load(f)
    elif llama_no_just:
        results_dir = "llama_results_no_just_wrubric" if args.wrubric else "llama_results_no_just"
        with open(f"{results_dir}/test_results.json", "r", encoding="utf-8") as f:
            pred_results = json.load(f)
    else:
        results_dir = f"{prefix}_results/results_{result_suffix}"
        with open(f"{results_dir}/classification_results.json", "r", encoding="utf-8") as f:
            pred_results = json.load(f)
        
        # check if ft_mode
        if ft_mode:
            # load bert_results 
            with open("bert_results/test_results.json", "r", encoding="utf-8") as f:
                bert_results = json.load(f)
            # delete the essays in pred_results that are not in bert_results
            pred_results = {k: v for k, v in pred_results.items() if k in bert_results}
            
            # create new results dir 
            results_dir = f"{results_dir}_ft_mode"
            os.makedirs(results_dir, exist_ok=True)

    # prepare a dataframe to store the results
    results = []

    subskill_map = {
        '2.1': '2.1: Synthesizing Multiple Sources',
        '2.2': '2.2: Evaluating Evidence Strength',
        '3.1': '3.1: Using Counterarguments',
        '3.2': '3.2: Using Facts and Opinions',
        '4.1': '4.1: Drawing Conclusions',
        '4.2': '4.2: Using Logical Fallacies'
    }

    for _, row in gt_df.iterrows():
        essay_id = row['essay_id']
        if essay_id not in pred_results:
            continue
        pred_rubric = pred_results[essay_id]
        for subskill, gt_label in row.items():
            if 'Subskill' not in subskill:
                continue
            subskill_number = subskill.split('Subskill ')[-1]
            subskill_name = subskill_map.get(subskill_number, subskill)
            pred_label = pred_rubric.get(subskill_name, {}).get('label', None)
            if pred_label not in ['Few-shot Example', 'Parsing Error'] and pred_label is not None:
                results.append({
                    'Essay ID': essay_id,
                    'Subskill': subskill,
                    'Ground Truth': gt_label.lower().strip(),
                    'Predicted': pred_label.lower().strip()
                })

    results_df = pd.DataFrame(results)

    # save the results to a CSV file
    results_df.to_csv(f"{results_dir}/classification_detailed_results.csv", index=False)

    # Process subskill metrics
    subskill_metrics = process_subskill_metrics(results_df, result_suffix, prefix, results_dir)
    
    # Save subskill metrics to CSV
    metrics_df = pd.DataFrame.from_dict(subskill_metrics, orient='index')
    # Round all float columns to 2 decimal places before saving
    metrics_df = metrics_df.round(2)
    metrics_df.to_csv(f"{results_dir}/subskill_metrics.csv")

    print("\n" + "=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)

    # calculate overall accuracy
    overall_accuracy = (results_df['Ground Truth'] == results_df['Predicted']).mean()
    
    # calculate overall RMSE
    overall_rmse = calculate_rmse(results_df['Ground Truth'], results_df['Predicted'])
    
    # calculate overall Krippendorff's alpha
    overall_alpha = calculate_krippendorff_alpha(results_df['Ground Truth'], results_df['Predicted'])
    
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Overall RMSE: {overall_rmse:.4f}")
    print(f"Overall Krippendorff's α: {overall_alpha:.4f}")

    # classification report
    print("\nClassification Report:")
    overall_report = classification_report(results_df['Ground Truth'], results_df['Predicted'], zero_division=0)
    print(overall_report)

    # confusion matrix
    print("\nConfusion Matrix:")
    overall_cm = confusion_matrix(results_df['Ground Truth'], results_df['Predicted'])
    print(overall_cm)
    
    # Save overall results using reusable function
    save_overall_results(results_dir, overall_accuracy, overall_rmse, overall_alpha, len(results_df),
                        overall_report, overall_cm, subskill_metrics)
    
    overall_file = f"{results_dir}/subskill_wise_results/overall.txt"
    print(f"\nOverall results saved to: {overall_file}")

if __name__ == "__main__":
    analyze_results()