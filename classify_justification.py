"""
Code for zero and few shot classification of student essays using OpenAI models
"""

from tqdm import tqdm
from prompt_llm_utils import construct_prompt_message, prompt_openai
import pandas as pd
import os
import time
import json
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def get_label_and_justification(response):
    """
    Extract the label and justification from the model's response.
    **Output format:**  
    - First line: `Label - <number>: <label name>`  
    - Second line: `Justification - <your justification>`  
    Return a tuple (label, justification)
    """

    # Handle None or empty responses
    if not response or response.strip() == "":
        print("WARNING: Empty or None response received!")
        return None, None

    lines = response.strip().split("\n")
    label = None
    justification = None

    for line in lines:
        line = line.strip()  # Remove any leading/trailing whitespace

        if line.startswith("Label -"):
            # Extract everything after "Label - <number>:"
            if ":" in line:
                label = line.split(":", 1)[1].strip()
            else:
                # Fallback: extract everything after "Label -"
                label = line[7:].strip()  # 7 is length of "Label -"

        elif line.startswith("Justification -"):
            # Extract everything after "Justification -"
            if ":" in line:
                justification = line.split(":", 1)[1].strip()
            else:
                # Fallback: extract everything after "Justification -"
                justification = line[15:].strip()  # 15 is length of "Justification -"

    return label, justification


def get_few_shot_text(few_shot_cats, gt_df):
    """
    Get few-shot example text for the given subskill
    """
    few_shot_text = "### Few-shot Examples\n"
    for cctr, (cat, essay_id) in enumerate(few_shot_cats.items()):
        if essay_id is not None:
            essay_row = gt_df[gt_df["essay_id"] == essay_id]
            if not essay_row.empty:
                essay_text = essay_row["essay_text"].values[0]
                # add to few shot examples text
                # upper case first letter of each space separated word in cat
                cat = " ".join([word.capitalize() for word in cat.split()])
                few_shot_text += (
                    f"#### Essay\n{essay_text}\n#### Output\nLabel - {cctr}: {cat}\n\n"
                )
    return few_shot_text


def process_essay_subskill(
    essay_id,
    essay_text,
    rubric_row,
    system_prompt,
    model_name,
    max_tokens,
    few_shot,
    few_shot_example_dict,
    few_shot_example_texts,
    justification_only,
    gt_label=None,
):
    """
    Process a single subskill for an essay and return the result.
    """
    subskill = rubric_row["Subskill"]

    # skip if the essay is already in few shot examples
    if (
        few_shot
        and essay_id
        in few_shot_example_dict[subskill.split(":")[0].strip()].values()
    ):
        label, justification = (
            "Few-shot Example",
            "This essay is used as a few-shot example.",
        )
    else:
        # read the user prompt (fresh)
        if justification_only:
            user_prompt_path = "prompts/ground_truth_user.txt"
        else:
            user_prompt_path = "prompts/user.txt"
        with open(user_prompt_path, "r", encoding="utf-8") as file:
            user_prompt = file.read().strip()

        # get information from the rubric row
        definition = rubric_row["Definition"]
        not_applicable = rubric_row["Not Applicable"]
        below_emerging = rubric_row["Below Emerging"]
        emerging = rubric_row["Emerging"]
        expanding = rubric_row["Expanding"]
        exemplifying = rubric_row["Exemplifying"]
        if few_shot:
            few_shot_examples = few_shot_example_texts.get(subskill, "")
        else:
            few_shot_examples = ""
        # fill the template values in the user prompt
        if justification_only:
            user_prompt = user_prompt.format(
                subskill=subskill,
                definition=definition,
                not_applicable=not_applicable,
                below_emerging=below_emerging,
                emerging=emerging,
                expanding=expanding,
                exemplifying=exemplifying,
                few_shot_examples=few_shot_examples,
                essay=essay_text,
                gt_label=gt_label,
            )
        else:
            user_prompt = user_prompt.format(
                subskill=subskill,
                definition=definition,
                not_applicable=not_applicable,
                below_emerging=below_emerging,
                emerging=emerging,
                expanding=expanding,
                exemplifying=exemplifying,
                few_shot_examples=few_shot_examples,
                essay=essay_text,
            )

        # construct the prompt message
        prompt_messages = construct_prompt_message(system_prompt, user_prompt)

        # prompt the model
        response = prompt_openai(
            prompt_messages,
            model=model_name,
            max_tokens=max_tokens,
            temperature=1.0,
            top_p=0.95,
        )
        # process the response
        try:
            label, justification = get_label_and_justification(response)
        except Exception as e:
            label, justification = "Parsing Error", "Parsing Error"

    time.sleep(1)
    return subskill, {"label": label, "justification": justification}


def process_single_essay(
    row,
    rubric_df,
    system_prompt,
    model_name,
    max_tokens,
    few_shot,
    few_shot_example_dict,
    few_shot_example_texts,
    prefix,
    result_suffix,
    write_lock,
    expected_subskills,
    justification_only,
    output_filename,
):
    """
    Process all subskills for a single essay with synchronous output dumping.
    Only processes missing subskills if essay is partially complete.
    """
    essay_id = row["essay_id"]
    essay_text = row["essay_text"]

    # Read current results to check what's already done
    result_file = f"{prefix}_results/results_{result_suffix}/{output_filename}"
    with write_lock:
        if os.path.exists(result_file):
            with open(result_file, "r", encoding="utf-8") as f:
                essay_results = json.load(f)
        else:
            essay_results = {}
    
    # Determine which subskills are already completed for this essay
    completed_subskills = set(essay_results.get(essay_id, {}).keys())
    
    # iterate over the rubric and process each subskill
    rubric_results = {}
    for rindex, rubric_row in rubric_df.iterrows():
        subskill = rubric_row["Subskill"]
        
        # Skip if this subskill is already completed
        if subskill in completed_subskills:
            continue
        
        # get ground truth label for justification_only mode
        gt_label = None
        if justification_only:
            # extract subskill number
            subskill_number = subskill.split(":")[0].strip()
            # construct column name as it appears in gt_df
            gt_column_name = f"Subskill {subskill_number}"
            gt_label = row[gt_column_name]
        
        subskill, result = process_essay_subskill(
            essay_id,
            essay_text,
            rubric_row,
            system_prompt,
            model_name,
            max_tokens,
            few_shot,
            few_shot_example_dict,
            few_shot_example_texts,
            justification_only,
            gt_label,
        )
        rubric_results[subskill] = result

    # Dump all results for this essay ONCE after all subskills are processed
    if rubric_results:  # Only dump if we processed at least one subskill
        with write_lock:
            # Read current results freshly to avoid race conditions
            if os.path.exists(result_file):
                with open(result_file, "r", encoding="utf-8") as f:
                    current_results = json.load(f)
            else:
                current_results = {}

            # Ensure the essay_id exists in current_results
            if essay_id not in current_results:
                current_results[essay_id] = {}
            
            # Update with ALL the new subskill results for this essay
            current_results[essay_id].update(rubric_results)

            # Write back to file
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(current_results, f, ensure_ascii=False, indent=4)

    return essay_id, rubric_results


def parse_args():
    parser = argparse.ArgumentParser(description="Classification of student essays")
    parser.add_argument(
        "--few_shot", action="store_true", help="Perform few shot prompting"
    )
    parser.add_argument(
        "--model_choice",
        type=int,
        default=0,
        help="Model Choice: 0 for gpt-4.1, 1 for gpt-5, 2 for gpt-5-mini",
    )
    parser.add_argument(
        "--justification_only",
        action="store_true",
        help="Peform justification generation given ground truth labels"
    )
    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()
    few_shot = args.few_shot
    model_choice = args.model_choice
    justification_only = args.justification_only

    # add model prefix
    if model_choice == 0:
        prefix = "gpt-4.1"
        model_name = "gpt-4.1"
        max_tokens = 500
    elif model_choice == 1:
        prefix = "gpt-5"
        model_name = "gpt-5"
        max_tokens = 3000
    elif model_choice == 2:
        prefix = "gpt-5-mini"
        model_name = "gpt-5-mini"
        max_tokens = 2000

    if justification_only:
        result_suffix = "justification_only"
    elif few_shot:
        result_suffix = "few_shot"
    else:
        result_suffix = "zero_shot"

    # read the rubrc
    rubric_df = pd.read_csv("rubric.csv")

    # read the ground truth data
    gt_df = pd.read_excel("combined_essays.xlsx")
    # reset the column headers to be the first row of the dataframe
    gt_df.columns = gt_df.iloc[0]
    gt_df = gt_df[1:]

    # if few_shot, prepare few shot examples
    few_shot_example_texts = {}
    if few_shot:
        # open few shot examples json
        with open("few_shot_examples.json", "r") as f:
            few_shot_example_dict = json.load(f)

        # iterate over the rubric to get few shot examples
        for rindex, rubric_row in rubric_df.iterrows():
            subskill = (
                rubric_row["Subskill"].split(":")[0].strip()
            )  # Extract subskill number, e.g., '2.1'
            few_shot_cats = few_shot_example_dict.get(subskill, {})
            few_shot_text = get_few_shot_text(few_shot_cats, gt_df)
            few_shot_example_texts[rubric_row["Subskill"]] = few_shot_text

        # dump few_shot_example_texts for verification
        with open("few_shot_example_texts.json", "w") as f:
            json.dump(few_shot_example_texts, f, indent=4)

    # # consider only first 10 essays for testing
    # gt_df = gt_df.head(10)

    # read the system prompt
    if justification_only:
        system_prompt_path = "prompts/ground_truth_system.txt"
    else:
        system_prompt_path = "prompts/system.txt"

    with open(system_prompt_path, "r", encoding="utf-8") as file:
        system_prompt = file.read().strip()

    # result directory
    os.makedirs(f"{prefix}_results/results_{result_suffix}", exist_ok=True)

    # conditional output filename
    if justification_only:
        output_filename = "justification_results.json"
    else:
        output_filename = "classification_results.json"

    # check if output file exists and load already processed essays
    result_file = f"{prefix}_results/results_{result_suffix}/{output_filename}"
    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            essay_results = json.load(f)
    else:
        essay_results = {}

    # Get all expected subskills from the rubric
    expected_subskills = set(rubric_df["Subskill"].tolist())
    
    # Filter out already processed essays (those with all specific subskills completed)
    essays_to_process = []
    for index, row in gt_df.iterrows():
        essay_id = row["essay_id"]
        # Check if essay has all specific subskills processed
        if essay_id in essay_results:
            # Get the set of completed subskills for this essay
            completed_subskills = set(essay_results[essay_id].keys())
            # Check if all expected subskills are present
            if expected_subskills.issubset(completed_subskills):
                continue
        essays_to_process.append(row)

    print(
        f"Total essays: {len(gt_df)}, Already processed: {len(gt_df) - len(essays_to_process)}, To process: {len(essays_to_process)}"
    )

    # print the essays to process   
    print("Essays to process:", [row["essay_id"] for row in essays_to_process])

    # Process essays in parallel (10 at a time)
    write_lock = threading.Lock()
    max_workers = 10
    failed_essays = {}  # Track failed essays and their errors

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all essay processing tasks
        future_to_essay = {
            executor.submit(
                process_single_essay,
                row,
                rubric_df,
                system_prompt,
                model_name,
                max_tokens,
                few_shot,
                few_shot_example_dict if few_shot else {},
                few_shot_example_texts if few_shot else {},
                prefix,
                result_suffix,
                write_lock,
                expected_subskills,
                justification_only,
                output_filename,
            ): row["essay_id"]
            for row in essays_to_process
        }

        # Process completed tasks with progress bar
        for future in tqdm(
            as_completed(future_to_essay),
            total=len(essays_to_process),
            desc="Processing essays in parallel",
        ):
            essay_id = future_to_essay[future]
            try:
                result_essay_id, rubric_results = future.result()
            except Exception as e:
                failed_essays[essay_id] = str(e)
    
    # Print summary of failed essays at the end
    if failed_essays:
        print(f"\n{'='*60}")
        print(f"PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total essays to process: {len(essays_to_process)}")
        print(f"Successfully processed: {len(essays_to_process) - len(failed_essays)}")
        print(f"Failed essays: {len(failed_essays)}")
        print(f"\nFailed essay IDs:")
        for essay_id, error in failed_essays.items():
            print(f"  - {essay_id}: {error}")
        print(f"{'='*60}")
    else:
        print(f"\nAll {len(essays_to_process)} essays processed successfully!")


if __name__ == "__main__":
    main()
