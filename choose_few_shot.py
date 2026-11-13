"""
Choose few-shot examples for each subskill
"""

import os
import json
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(description="Choose few-shot examples for each subskill")
    parser.add_argument("--k", type=int, default=1, help="Number of examples to select per category (default: 1)")
    args = parser.parse_args()
    k = args.k
    # read the ground truth data
    gt_df = pd.read_excel("combined_essays.xlsx")
    # reset the column headers to be the first row of the dataframe
    gt_df.columns = gt_df.iloc[0]
    gt_df = gt_df[1:]

    # consider one example for each unique label in each subskill
    subskill_list = ["2.1", "2.2", "3.1", "3.2", "4.1", "4.2"]
    categories = [
        "not applicable",
        "below emerging",
        "emerging",
        "expanding",
        "exemplifying",
    ]
    few_shot_examples = {}
    for subskill in subskill_list:
        subskill_col = f"Subskill {subskill}"
        # lower case all entries in the subskill column
        gt_df[subskill_col] = gt_df[subskill_col].str.lower().str.strip()
        few_shot_examples[subskill] = {}
        for category in categories:
            category_rows = gt_df[gt_df[subskill_col] == category]
            if len(category_rows) > 0:
                num_examples = min(k, len(category_rows))
                selected_essays = category_rows.head(num_examples)["essay_id"].tolist()
                few_shot_examples[subskill][category] = selected_essays
                # print(f"Subskill {subskill}, Category '{category}': Selected {len(selected_essays)} examples")
            else:
                few_shot_examples[subskill][category] = []

    # save the few shot examples
    output_filename = f"few_shot_examples_{k}.json"
    with open(output_filename, "w") as f:
        json.dump(few_shot_examples, f, indent=4)
    print(f"\nSaved few-shot examples to {output_filename}")


if __name__ == "__main__":
    main()
