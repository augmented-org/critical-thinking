# Toward LLM-Supported Automated Assessment of Critical Thinking Subskills

Code and data for the paper:

**Toward LLM-Supported Automated Assessment of Critical Thinking Subskills**  
Marisa Peczuh, Nischal Ashok Kumar, Ryan Baker, Blair Lehman, Danielle Eisenberg, Caitlin Mills, Payu Wittawatolarn, Kushaan Naskar, Keerthi Chebrolu, Sudhip Nashi, Cadence Young, Brayden Liu, Sherry Lachman, Andrew Lan  
*Proceedings of the 19th International Conference on Educational Data Mining (EDM 2026)*, pp. 340–352  
DOI: [10.5281/zenodo.21039869](https://doi.org/10.5281/zenodo.21039869) · arXiv: [2510.12915](https://arxiv.org/abs/2510.12915)

---

## Overview

We study whether the subskills that underlie critical thinking can be measured automatically in student argumentative essays. We develop a coding rubric adapted from the *Skills for the Future* Critical Thinking Progression (3 skills, 6 subskills, 5 proficiency levels), human-code 500 essays, and compare three automated approaches:

1. **Zero-shot prompting** — GPT-5
2. **Few-shot prompting** — GPT-5 with in-context examples for each proficiency level
3. **Supervised fine-tuning** — Llama 3.1 8B Instruct (LoRA) and ModernBERT-base

We evaluate under two splits: an **essay-based split** (70/10/20, averaged over 5 random seeds) and a **subskill-based split** (train on two skills, test on the held-out third).

---

## Data Release

This repository releases the human-annotated dataset and rubric used in the paper.

| File | Description |
|---|---|
| `combined_essays.xlsx` | 500 human-coded essays. One row per essay with `essay_id`, `essay_text`, and a proficiency label for each subskill (`Subskill 2.1` … `Subskill 4.2`). The first row is a skill-group header row; the scripts handle this automatically. |
| `Critical Thinking Rubric for Essay Rating (EDM 2026).pdf` | The full coding rubric used by the human coders, with definitions, level descriptors, notes, and examples. |
| `rubric.csv` | Machine-readable version of the rubric (subskill name, definition, and descriptor for each proficiency level) used as model input by the scripts. |
| `justification_detailed_results.csv` | Synthetic GPT-4.1 justifications for each (essay, subskill, gold label) triple — 3,000 rows. Used as training targets for the Llama *with justification* variants. |
| `few_shot_examples_{1,3,5}.json` | Essay IDs used as in-context examples for few-shot prompting (k = 1, 3, 5 per proficiency level). These essays are excluded from few-shot evaluation. |

**Rubric (3 skills, 6 subskills):**

| Skill | Subskill |
|---|---|
| 2. Information Analysis | 2.1 Synthesizing Multiple Sources · 2.2 Evaluating Evidence Strength |
| 3. Argument Generation | 3.1 Using Counterarguments · 3.2 Using Facts and Opinions |
| 4. Logical Reasoning | 4.1 Drawing Conclusions · 4.2 Using Logical Fallacies |

**Proficiency levels:** Not Applicable · Below Emerging · Emerging · Expanding · Exemplifying

**Essay source.** The essays are a subset of the PERSUADE 2.0 corpus (grades 6–12, 15 prompts), released through the Kaggle Feedback Prize competition. If you use the essays, please also cite PERSUADE 2.0 (Crossley et al.).

All data files must be in the root of the repository (the scripts read them from the working directory).

---

## Setup

```bash
conda env create -f env.yml
conda activate torchnish
```

Environment variables:
- `OPENAI_API_KEY` — for GPT-5 prompting and GPT-4.1 justification generation
- A Hugging Face token with access to `meta-llama/Llama-3.1-8B-Instruct` (e.g., `huggingface-cli login`)

---

## Repository Structure

```
.
├── combined_essays.xlsx              # Human-coded essays (data release)
├── Critical Thinking Rubric for Essay Rating (EDM 2026).pdf  # Full rubric (data release)
├── rubric.csv                        # Machine-readable rubric used by the scripts
├── justification_detailed_results.csv# GPT-4.1 synthetic justifications
├── few_shot_examples_{1,3,5}.json    # Few-shot example essay IDs
├── env.yml                           # Conda environment
├── prompts/
│   ├── system.txt                    # System prompt for GPT-5 prompting
│   ├── user.txt                      # User prompt for GPT-5 prompting
│   ├── ground_truth_system.txt       # System prompt for justification generation
│   └── ground_truth_user.txt         # User prompt for justification generation
├── prompt_llm_utils.py               # OpenAI API prompting utilities
├── llama_data.py                     # Data loading and train/val/test splits
├── classify_justification.py         # Step 1: Generate GPT-4.1 justifications
├── choose_few_shot.py                # Step 2a: Select few-shot examples
├── classify.py                       # Step 2b: GPT-5 zero-/few-shot prompting
├── llama_train_wrubric.py            # Step 3: Fine-tune Llama 3.1 8B (with rubric descriptors)
├── llama_train.py                    # Step 3: Fine-tune Llama 3.1 8B (without rubric descriptors)
├── modernbert_classify_wrubric.py    # Step 4: Fine-tune ModernBERT (with rubric descriptors)
├── modernbert_classify.py            # Step 4: Fine-tune ModernBERT (without rubric descriptors)
├── analyze_results.py                # Step 5: Compute metrics
└── misclassification_error_analysis.py # Step 6: Error analysis
```

---

## Pipeline

### Step 1: Generate Justifications (optional — output already provided)

No human rationales exist for the labels, so we generate a short justification for each gold label with GPT-4.1. These serve as training targets for the Llama *with justification* variants (the model outputs a justification, then the label).

```bash
python classify_justification.py --model_choice 0 --justification_only
```

Output: `gpt-4.1_results/results_justification_only/justification_results.json`. The flattened version used for training is released as `justification_detailed_results.csv`.

---

### Step 2: GPT-5 Prompting

Each essay is paired with one target subskill; the model outputs a label (0–4) and a justification.

**Zero-shot:**
```bash
python classify.py --model_choice 1
```

**Few-shot** (k examples per proficiency level; k = 3 is reported in the main results):
```bash
python choose_few_shot.py --k 3        # writes few_shot_examples_3.json (already provided)
python classify.py --model_choice 1 --few_shot --k 3
```

We also tested k = 1 and k = 5 (`--k 1`, `--k 5`).

Results are written to `gpt-5_results/results_{zero_shot,few_shot_k3}/classification_results.json`. Prompting is run on all 500 essays; Step 5 evaluates it on the same 5 test splits used for fine-tuning.

---

### Step 3: Fine-tune Llama 3.1 8B Instruct

Input: subskill name, definition, (optional) rubric descriptors, and the essay. Output: the label (optionally preceded by a justification). LoRA (r = 32, α = 64, dropout = 0.05), AdamW, 3 epochs, batch size 4, gradient accumulation 4, max sequence length 1,536. These are the script defaults.

**Essay-based split** (70/10/20, seeds 47, 123, 456, 789, 999):

| Variant (Table 5) | Command | Output dir |
|---|---|---|
| With rubric, no justification **(best)** | `python llama_train_wrubric.py --multiple_splits --no_just` | `llama_results_no_just_wrubric_splits/` |
| With rubric, with justification | `python llama_train_wrubric.py --multiple_splits` | `llama_results_wrubric_splits/` |
| Without rubric, no justification | `python llama_train.py --multiple_splits --no_just` | `llama_results_no_just_splits/` |
| Without rubric, with justification | `python llama_train.py --multiple_splits` | `llama_results_splits/` |

**Subskill-based split** (train on two skills, test on the held-out skill; 90/10 train/val essay split):

| Variant | Command | Output dir |
|---|---|---|
| With rubric, with justification (Table 7) | `python llama_train_wrubric.py --gen_subskills` | `llama_results_wrubric_gen_subskills/` |
| With rubric, no justification | `python llama_train_wrubric.py --gen_subskills --no_just` | `llama_results_no_just_wrubric_gen_subskills/` |
| Without rubric, with justification | `python llama_train.py --gen_subskills` | `llama_results_gen_subskills/` |

Add `--use_wandb` to any command to log to Weights & Biases.

---

### Step 4: Fine-tune ModernBERT-base

Classification head over the [CLS] representation; focal loss, AdamW, 6 epochs with early stopping, max input length 1,536.

```bash
# With rubric descriptors (Table 4)
python modernbert_classify_wrubric.py --multiple_splits     # -> modernbert_results_wrubric_splits/

# Without rubric descriptors
python modernbert_classify.py --multiple_splits             # -> modernbert_results_splits/
```

---

### Step 5: Evaluate

`analyze_results.py` computes Accuracy, RMSE, Macro F1, Weighted F1, and Krippendorff's α (overall and per subskill), averaged over the 5 seeds for the essay-based split.

**Essay-based split (Tables 4, 5, 6):**
```bash
# Llama variants
python analyze_results.py --llama_no_just --wrubric --split   # best model; per-subskill results = Table 6
python analyze_results.py --llama --wrubric --split
python analyze_results.py --llama_no_just --split
python analyze_results.py --llama --split

# ModernBERT
python analyze_results.py --modernbert --wrubric --split
python analyze_results.py --modernbert --split

# GPT-5, evaluated on the same 5 test splits
python analyze_results.py --split --ft_mode --model_choice 1
python analyze_results.py --split --ft_mode --model_choice 1 --few_shot --k 3
```

Each writes `overall_results/overall_statistics.txt` and `overall_results/subskill_statistics.csv` inside the corresponding results directory.

**Subskill-based split (Table 7):**
```bash
python analyze_results.py --gen_subskills --llama --wrubric
python analyze_results.py --gen_subskills --llama_no_just --wrubric
python analyze_results.py --gen_subskills --llama
```

Writes `overall_results/gen_subskills_summary.{csv,txt}` inside the corresponding results directory.

**GPT-5 zero-shot on all 500 essays** (reference point for the subskill-based comparison):
```bash
python analyze_results.py --model_choice 1
```

---

### Step 6: Error Analysis (Subskill 4.2)

Confusion analysis for *Using Logical Fallacies* with the Llama model (with rubric, with justification), run per seed:

```bash
for rs in 47 123 456 789 999; do
  python misclassification_error_analysis.py \
      --results-file llama_results_wrubric_splits/rstate_${rs}/test_results.json \
      --gt-file combined_essays.xlsx \
      --only-subskill 4.2
done
```

The paper reports the misclassification counts summed across the five seeds.

---

## Mapping to Paper

| Paper | Script | Flags |
|---|---|---|
| Table 4 — Llama 8B fine-tune | `llama_train_wrubric.py` | `--multiple_splits --no_just` |
| Table 4 — ModernBERT fine-tune | `modernbert_classify_wrubric.py` | `--multiple_splits` |
| Table 4 — GPT-5 zero-shot | `classify.py` | `--model_choice 1` |
| Table 4 — GPT-5 few-shot | `classify.py` | `--model_choice 1 --few_shot --k 3` |
| Table 5 — rubric × justification ablation | `llama_train_wrubric.py` / `llama_train.py` | `--multiple_splits` ± `--no_just` |
| Table 6 — per-subskill results | `analyze_results.py` | `--llama_no_just --wrubric --split` |
| Table 7 — subskill-based split | `llama_train_wrubric.py` | `--gen_subskills` |
| Error analysis (4.2) | `misclassification_error_analysis.py` | `--only-subskill 4.2` |

---

## Citation

If you use this code, data, or rubric, please cite:

```bibtex
@inproceedings{2026.EDM.full-papers.174,
 abstract = {As the world becomes increasingly saturated with AI-generated content, disinformation, and algorithmic persuasion, critical thinking – the capacity to evaluate evidence, detect unreliable claims, and exercise independent judgment – is becoming a defining human skill. Developing critical thinking skills through timely assessment and feedback is crucial; however, there has not been extensive work in educational data mining on defining, measuring, and supporting critical thinking. In this paper, we investigate the feasibility of meas-uring “subskills” that underlie critical thinking. We ground our work in an authentic task where students operationalize critical thinking by writing argumentative essays. We devel-oped a coding rubric based on an established skills progres-sion and completed human coding for a corpus of student es-says. We then evaluated three distinct approaches to automat-ed scoring: zero-shot prompting, few-shot prompting, and su-pervised fine-tuning, implemented across three large language models (GPT-5, Llama 3.1 8B, and ModernBERT). Fine-tuning Llama 3.1 8B achieved the best results and demonstrated par-ticular strength on subskills with highly separable proficiency levels with balanced labels across levels, while lower perfor-mance was observed for subskills that required detection of subtle distinctions between proficiency levels or imbalanced labels. Our exploratory work represents an initial step toward scalable assessment of critical thinking skills across authentic educational contexts. Future research should continue to com-bine automated critical thinking assessment with human vali-dation to more accurately detect and measure dynamic, higher-order thinking skills.},
 address = {Seoul, Republic of Korea},
 author = {Marisa Peczuh and Nischal Ashok Kumar and Ryan Baker and Blair Lehman and Danielle Eisenberg and Caitlin Mills and Payu Wittawatolarn and Kushaan Naskar and Keerthi Chebrolu and Sudhip Nashi and Cadence Young and Brayden Liu and Sherry Lachman and Andrew Lan},
 booktitle = {Proceedings of the 19th International Conference on Educational Data Mining},
 doi = {10.5281/zenodo.21039869},
 editor = {Anthony Botelho and Maria Mercedes T. Rodrigo and Adish Singla and Hiroaki Ogata and Hyojeong So and Young Hoan Cho},
 isbn = {978-1-7336736-7-9},
 month = {June},
 pages = {340--352},
 publisher = {International Educational Data Mining Society},
 title = {Toward LLM-supported Automated Assessment of Critical Thinking Subskills},
 year = {2026}
}
```
