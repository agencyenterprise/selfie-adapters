# Training Data

This directory contains training data for SelfIE adapters.

## Included Data

- `goodfire_8b_sae_labels.json.gz` - SAE decoder vectors with auto-interpretability labels from Goodfire's Llama 3.1 8B layer 19 SAE

### Setup

**Important:** Decompress this file before training:

```bash
cd data
gunzip goodfire_8b_sae_labels.json.gz
```

This creates `goodfire_8b_sae_labels.json` which is referenced by the training configs (e.g., `training/configs/scalar_affine_8b_goodfire.yaml`).

## Data Format

Training data is a JSON file containing an array of dataset sections:

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
      {"index": 0, "labels": ["label 1", "label 2"], "split": "train"},
      {"index": 1, "labels": ["another label"], "split": "val"}
    ]
  }
]
```

The `filename` field points to a `.pt` file containing a tensor of shape `(num_vectors, model_dim)` where each row is an SAE decoder vector.

## Preparing Your Own Training Data

Use the `data_prep/prepare_sae_training_data.py` script to prepare training data from any SAELens-compatible SAE:

### LlamaScope SAEs (auto-downloads labels from Neuronpedia)

```bash
python data_prep/prepare_sae_training_data.py \
    --release llama_scope_lxr_8x \
    --sae-id l19r_8x \
    --output-dir data/ \
    --output-name llamascope_l19_32k
```

### Goodfire 70B SAE (auto-downloads labels from Neuronpedia)

```bash
python data_prep/prepare_sae_training_data.py \
    --release goodfire-llama-3.3-70b-instruct \
    --sae-id layer_50 \
    --output-dir data/ \
    --output-name goodfire_70b_l50
```

### Goodfire 8B SAE (requires labels JSON - not on Neuronpedia)

```bash
python data_prep/prepare_sae_training_data.py \
    --release goodfire-llama-3.1-8b-instruct \
    --sae-id layer_19 \
    --output-dir data/ \
    --output-name goodfire_8b_l19 \
    --labels-json path/to/goodfire_labels.json
```

### Supported SAELens Releases

| Release | SAE ID format | Width | Labels Source |
|---------|---------------|-------|---------------|
| `llama_scope_lxr_8x` | `l{layer}r_8x` | 32k | Neuronpedia S3 |
| `llama_scope_lxr_32x` | `l{layer}r_32x` | 131k | Neuronpedia S3 |
| `llama_scope_lxm_8x` | `l{layer}m_8x` | 32k | Neuronpedia S3 |
| `llama_scope_lxm_32x` | `l{layer}m_32x` | 131k | Neuronpedia S3 |
| `goodfire-llama-3.3-70b-instruct` | `layer_50` | 65k | Neuronpedia S3 |
| `goodfire-llama-3.1-8b-instruct` | `layer_19` | 65k | Manual JSON (not on Neuronpedia) |

The script will:
1. Load the SAE from SAELens
2. Fetch auto-interpretability labels (from Neuronpedia S3 for LlamaScope and Goodfire 70B, or from provided JSON for Goodfire 8B)
3. Create train/val splits (default 90/10)
4. Save a `.pt` file with decoder vectors and a `_labels.json` file in training format

### Manual Data Preparation

Alternatively, prepare data manually:

1. **Vectors file (.pt):** SAE decoder vectors as a tensor of shape `(num_sae_features, model_dim)`
2. **Labels JSON:** The format shown above, with labels for each vector and train/val splits

The vectors can be extracted from any SAELens-compatible SAE using `sae.W_dec` (decoder weights).

See the paper or the main README for more details on training data requirements.
