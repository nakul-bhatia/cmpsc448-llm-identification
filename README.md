# Who Wrote It? Identifying LLMs from Their Responses

**Author:** Nakul Bhatia  
**Course:** CMPSC 448  
**Models:** CNN and LSTM

[Read the project report](CMPSC448_Midterm_Report_Nakul_Bhatia.pdf)

## Overview

This project investigates whether neural networks can identify which LLM family generated a response: GPT, Claude, or Gemini.

Two research questions are evaluated:

- **RQ1:** Can the LLM family be identified from its response alone?
- **RQ2:** Does including the user's prompt improve identification?

Both CNN and LSTM classifiers are evaluated using prompt-only, response-only, and combined inputs.

## Dataset

Source: [LMArena arena-human-preference-55k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k).

The balanced sample contains 1,500 examples:
- GPT: 500
- Claude: 500
- Gemini: 500

Only single-turn interactions are retained. Empty examples and duplicate family/prompt/response triples are removed. Prompts must contain at least 3 words and responses at least 30 words. PaLM models are excluded.

Normalized matching prompts are grouped into the same partition to reduce leakage:

| Partition | Examples |
|---|---:|
| Training | 1,196 |
| Validation | 152 |
| Test | 152 |

The 20,000-token vocabulary is built using training text only.

## Models and Training

Both models use trainable 128-dimensional word embeddings.

- **CNN:** Convolution widths 3, 4, and 5; 64 filters per width; ReLU; global max pooling; dropout 0.5; three-class output.
- **LSTM:** One unidirectional layer with 64 hidden units; packed sequences to exclude padding; dropout 0.5; three-class output.

Shared settings:
- Maximum sequence length: 256 tokens
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32
- Epochs: 10
- Random seed: 42
- Loss: cross-entropy
- Checkpoint selection: highest validation accuracy

LSTM gradient norms are clipped to 1.0. The best validation checkpoint is loaded before test evaluation.

## Results

| Model | Input | Test Accuracy | Macro F1 |
|---|---|---:|---:|
| CNN | Prompt only | 34.87% | 0.3458 |
| CNN | Response only | 72.37% | 0.7260 |
| CNN | Prompt + response | 75.00% | 0.7523 |
| LSTM | Prompt only | 38.16% | 0.3817 |
| LSTM | Response only | 54.61% | 0.5475 |
| LSTM | Prompt + response | 50.00% | 0.5010 |

Macro F1 values above are calculated from the recorded confusion matrices. The output-only entries in all_results.csv contain rounded historical values.

Response text provides most of the useful classification signal in these experiments. Adding prompts helped the CNN slightly but reduced LSTM performance.

Combined inputs reserve up to 64 prompt tokens plus a separator, leaving less space for the response. Therefore, the comparison also reflects different response truncation. These are single-seed results, not evidence of a universal advantage for either architecture.

## Installation

Install Python and the required packages:

```bash
python3 -m pip install torch pandas numpy scikit-learn datasets
```

Package versions are not pinned. Results may vary across hardware and software versions.

## Run Using the Included Data

Open a terminal in the repository directory. Keep the CSV files and vocab.json alongside the Python scripts.

```bash
python3 train_cnn.py
python3 train_lstm.py
python3 run_rq2.py
```

The scripts automatically select CUDA, Apple MPS, or CPU, depending on availability. They train the models and save checkpoints locally.

## Rebuild the Dataset and Vocabulary

To download and prepare the source data again:

```bash
python3 prepare_data.py
python3 split_data.py
python3 preprocess.py
```

These commands overwrite the prepared CSV files and vocabulary. The source dataset revision is not pinned, so a future download may differ from the included data.

## Files

| File | Purpose |
|---|---|
| prepare_data.py | Download, filter, and balance the dataset |
| split_data.py | Create grouped training, validation, and test splits |
| preprocess.py | Build the training-only vocabulary |
| train_cnn.py | Train and evaluate the response-only CNN |
| train_lstm.py | Train and evaluate the response-only LSTM |
| run_rq2.py | Run prompt-only and combined-input experiments |
| llm_dataset.csv | Full balanced sample |
| train.csv, val.csv, test.csv | Prepared data partitions |
| vocab.json | Token-to-ID vocabulary |
| rq2_results.csv | Results from the four RQ2 runs |
| all_results.csv | RQ2 results plus historical RQ1 values |
| CMPSC448_Midterm_Report_Nakul_Bhatia.pdf | Full report and analysis |

**Reproduction note:** run_rq2.py appends hard-coded historical output-only results. It does not automatically import new results from train_cnn.py or train_lstm.py. Update those entries when reporting a new run.

Model checkpoint files are not included in this repository; running the training scripts recreates them.

## Acknowledgments

The source dataset is provided by LMArena; its dataset card lists Apache-2.0.

ChatGPT assisted with code development, debugging, interpretation, and report drafting. The author is responsible for understanding and verifying the submitted work.
