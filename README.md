# SelfIE Adapters

Training and evaluation code for SelfIE (Self-Interpretation via Embedding) adapters, which enable language models to describe the meaning of their own internal activations.

## Overview

This repository contains code for:
- **Training** lightweight adapters that map hidden state activations to soft token embeddings
- **Evaluating** trained adapters on generation scoring, embedding retrieval, and bridge entity extraction tasks
- **Preparing** training and evaluation datasets

## Repository Structure

```
├── selfie_adapters/          # Shared adapter architecture code
│   ├── projection.py         # Projection module implementations
│   ├── inference.py          # Lightweight inference utilities
│   └── sae_utils.py          # SAE and model loading utilities
│
├── training/                 # Training infrastructure
│   ├── train.py              # Main training entry point
│   ├── train_modal.py        # Modal wrapper for cloud training
│   ├── model.py              # Language model wrapper with soft token injection
│   ├── trainer.py            # Training loop with logging
│   ├── config.py             # Configuration dataclasses
│   ├── data.py               # Dataset loading and mixing
│   └── configs/              # Example YAML configurations
│
├── evals/                    # Evaluation scripts
│   ├── generation_scoring/   # SAE latent generation and scoring (includes Modal wrapper)
│   ├── embedding_retrieval/  # Embedding-based topic retrieval eval
│   └── bridge_entity/        # TwoHopFact bridge entity extraction (includes Modal wrapper)
│
└── data_prep/                # Dataset preparation
    ├── wikipedia_topics/     # Wikipedia vital articles processing
    │   ├── extract_wikipedia_vectors.py  # Extract activation vectors for training
    │   └── dataset_generation/           # Anthropic Batch API pipeline for dataset creation
    └── twohopfact_filtering/ # TwoHopFact dataset filtering
```

## Adapter Architectures

The paper evaluates several projection architectures, ordered by performance:

| Architecture | Parameters | Description |
|-------------|------------|-------------|
| `scalar_affine_plus_low_rank` | d + 2dr + 1 | Best performing: f(x) = scale·x + x·UV^T + bias |
| `scalar_affine` | d + 1 | Strong baseline: f(x) = scale·x + bias |
| `low_rank_only` | 2dr + d | Pure low-rank: f(x) = x·UV^T + bias |
| `scale_only` | 1 | Minimal: f(x) = scale·x |
| `full_rank` | d² + d | Overfits; not recommended |

Where d = model dimension (4096 for Llama 3.1 8B) and r = rank.

**Identity baseline (0 parameters):** For the identity transformation f(x) = x used as a baseline in the paper, use `scale_only` with `init_scale: 1.0` and freeze the scale parameter, or simply use the untrained baseline in evaluation scripts.

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- Python >= 3.10
- PyTorch >= 2.0
- transformers >= 4.35
- sae-lens >= 4.0

## Data Files

Some data files are stored compressed to reduce repository size. Decompress before use:

```bash
# Training data (SAE decoder vectors with labels from Goodfire)
gunzip data/goodfire_8b_sae_labels.json.gz

# Wikipedia vital articles titles (for topic vector extraction)
gunzip data_prep/wikipedia_topics/vital_articles_level5.json.gz
```

## Quick Start

### Training an Adapter

```bash
# First, decompress the training data
gunzip data/goodfire_8b_sae_labels.json.gz

cd training
python train.py --config configs/scalar_affine_8b_goodfire.yaml
```

### Loading a Trained Adapter for Inference

```python
from selfie_adapters import load_adapter

# Load trained adapter
adapter = load_adapter("checkpoint.pt")

# Transform SAE decoder vectors to soft tokens
soft_tokens = adapter.transform(sae_vectors)

# Get adapter metadata
print(adapter.get_metadata())
```

### Running Evaluations

#### Generation Scoring (SAE Latents)

Evaluates label quality by checking if generated text containing the label actually activates the corresponding SAE latent.

```bash
cd evals/generation_scoring

# With trained adapter (generates labels then scores them)
python run_eval.py --config-file configs/example_label_generator.json

# With existing labels only (scores pre-generated labels)
python run_eval.py --config-file configs/example_label_dataset.json
```

#### Embedding Retrieval (Wikipedia Topics)

Evaluates label quality by checking if embeddings of generated labels can retrieve the correct topic from ~50k Wikipedia articles. This is an in-distribution eval for adapters trained on Wikipedia topic vectors.

**Workflow:**

1. **Generate labels** for Wikipedia topic vectors using a trained adapter (via generation_scoring in `label_generation_only` mode)
2. **Run retrieval eval** to measure recall@K

```bash
cd evals/embedding_retrieval

# Run skyline baselines (title->title and synthetic labels)
python topic_retrieval_eval.py

# To evaluate custom labels from a trained adapter:
# Use the evaluate_custom_labels() function with your generated labels
```

```python
from topic_retrieval_eval import TopicRetrievalIndex, evaluate_custom_labels, print_eval_summary

# Build or load the topic index
index = TopicRetrievalIndex()
index.build_or_load_index()

# Load your generated labels (one per topic, in order)
generated_labels = [...]  # List of strings from your adapter

# Evaluate
results = evaluate_custom_labels(index, generated_labels)
print_eval_summary(results, "Trained Adapter Labels")
```

#### Bridge Entity Extraction (TwoHopFact)

Tests whether SelfIE can extract implicit "bridge entities" from multi-hop reasoning activations.

**Prerequisites:**
1. **Generate the filtered dataset first:** Run the TwoHopFact filtering pipeline (see [TwoHopFact Filtering](#twohopfact-filtering) below). This filters the original TwoHopFact dataset to only include questions the model answers correctly, producing `filtered_dataset.json`.
2. For trained mode: provide a trained adapter checkpoint (train one using [Training](#training-an-adapter))

**Setup:**
```bash
cd evals/bridge_entity

# Copy or symlink the filtered dataset generated by the filtering pipeline
cp /path/to/filtered_data/filtered_dataset.json twohopfact_filtered.json

# For trained mode, also copy your adapter checkpoint
mkdir -p checkpoints
cp /path/to/your/checkpoint.pt checkpoints/scalar_affine_best.pt
```

**Running:**
```bash
# Untrained baseline (vanilla SelfIE with scaling only)
python run_selfie_bridge_extraction.py configs/example_untrained.json

# With trained adapter
python run_selfie_bridge_extraction.py configs/example_trained.json
```

The configs specify which question to analyze (`question_id`), layers to examine, and output directory for heatmaps.

## Training Data Format

Training data should be a JSON file with the following structure:

```json
[
  {
    "metadata": {
      "dataset_name": "goodfire_8b",
      "filename": "vectors.pt",
      "layer": 19,
      "source": "goodfire-llama-3.1-8b-instruct",
      "vector_type": "sae_decoder"
    },
    "vectors": [
      {"index": 0, "labels": ["descriptive label 1", "label 2"], "split": "train"},
      {"index": 1, "labels": ["another label"], "split": "val"}
    ]
  }
]
```

## Configuration

Training is configured via YAML files. Key parameters:

```yaml
projection:
  type: "scalar_affine_plus_low_rank"  # Architecture type
  normalize_input: true                 # L2 normalize inputs
  init_scale: 30.0                      # Initial scale value
  low_rank_rank: 64                     # Rank for low-rank component

training:
  learning_rate: 0.01
  num_epochs: 2
  batch_size: 80
```

See `training/configs/` for complete examples.

## Running on Modal

Both training and evaluation can be run on [Modal](https://modal.com) cloud infrastructure for access to A100 GPUs.

### Setup

```bash
# Install Modal CLI
pip install modal

# Authenticate
modal setup

# Create required secrets
modal secret create wandb-secret WANDB_API_KEY=your_key_here
modal secret create huggingface-token HF_TOKEN=your_token_here

# Upload training data to Modal volume
modal volume create sae-data
modal volume put sae-data data/goodfire_8b_sae_labels.json /goodfire_8b_sae_labels.json
```

### Training on Modal

```bash
cd training

# Run training with a config file
modal run train_modal.py --config configs/scalar_affine_8b_goodfire.yaml

# Override settings
modal run train_modal.py --config configs/scalar_plus_low_rank_8b.yaml --batch-size 32 --num-epochs 3

# Evaluation-only mode (for identity baseline)
modal run train_modal.py --config configs/identity_baseline.yaml --eval-only

# Download checkpoints after training
modal volume get selfie-adapter-training /checkpoints ./checkpoints/
```

### Generation Scoring on Modal

```bash
cd evals/generation_scoring

# Run with parallel shards for faster evaluation
modal run run_eval_modal.py --config-file configs/example_label_generator.json --num-parallel-instances 4

# Download results
modal volume get sae-eval-results / ./results/
```

### Bridge Entity Extraction on Modal

```bash
cd evals/bridge_entity
modal run run_modal.py --config-path configs/example_untrained.json
```

## Data Preparation

### Wikipedia Topic Dataset Generation

The paper uses a dataset of ~50k Wikipedia topics with prompts and alternate descriptions, published on HuggingFace as [`keenanpepper/fifty-thousand-things`](https://huggingface.co/datasets/keenanpepper/fifty-thousand-things). You can either download this dataset directly, or regenerate it using the scripts below.

**Prerequisites:**
- Set up your Anthropic API key: `export ANTHROPIC_API_KEY=your_key_here`
- Or create a `.env` file in the repo root with `ANTHROPIC_API_KEY=your_key_here`

**Generation Pipeline:**

The dataset is created through a multi-stage pipeline using the Anthropic Batch API:

```bash
cd data_prep/wikipedia_topics

# 1. Decompress the input titles
gunzip vital_articles_level5.json.gz

cd dataset_generation

# 2. Generate prompts and labels (run 4 times for better coverage, renaming outputs)
python generate_all.py
python check_batch.py  # Wait for completion, saves to outputs/generated_topics.json
# Rename output and repeat: mv outputs/generated_topics.json outputs/generated_topics_1.json

# 3. Merge multiple generation runs
python merge_topics.py  # Creates outputs/generated_topics_merged.json

# 4. (Optional) If prompts differ between runs, choose best prompts
python choose_best_prompts.py
python check_chosen_prompts.py
python merge_topics.py  # Re-run to use chosen prompts

# 5. Score label coherence
python score_coherence.py
python check_coherence_scores.py

# 6. Filter to high-coherence entries (score >= 9)
python filter_by_coherence.py 9

# 7. Create train/val splits
python create_jsonl_splits.py  # Creates final JSONL for HuggingFace
```

**Output:** The final dataset is saved to `outputs/wikipedia_vital_articles_level5_dataset.jsonl` with 90% train / 10% val split.

### Wikipedia Topic Vectors

To create contrastive topic vectors from Wikipedia article titles (for training):

1. Decompress the titles file:
```bash
cd data_prep/wikipedia_topics
gunzip vital_articles_level5.json.gz
```

2. Run the extraction script:
```bash
python extract_wikipedia_vectors.py \
    --titles-file vital_articles_level5.json \
    --output-vectors outputs/wikipedia_vectors_l19.pt \
    --output-metadata outputs/wikipedia_metadata_l19.json \
    --output-dataset outputs/wikipedia_contrastive_dataset.pt \
    --layer 19
```

This extracts residual stream activations for "Tell me about {title}." prompts and creates contrastive vectors by subtracting the mean.

### TwoHopFact Filtering

To filter the TwoHopFact dataset for questions the model answers correctly:

1. Start a vLLM server with Llama 3.1 8B:
```bash
cd data_prep/twohopfact_filtering
python start_vllm_server.py
# Or manually:
# python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Llama-3.1-8B-Instruct \
#     --port 8000
```

2. Run the filtering script:
```bash
python filter_dataset.py \
    --output-dir ./filtered_data \
    --batch-size 1000 \
    --vllm-url http://localhost:8000/v1
```

This filters for questions where the model correctly answers both the bridge entity (e2) and final answer (e3), excluding "shortcut" cases where the model guesses correctly without knowing the intermediate entity.

## Citation

If you use this code, please cite:

```bibtex
% PLACEHOLDER - This citation will be updated with final publication details
@inproceedings{selfie2026,
  title={Learning Self-Interpretation from Interpretability Artifacts: Training Lightweight Adapters on Vector-Label Pairs},
  author={TBD},
  booktitle={TBD},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
