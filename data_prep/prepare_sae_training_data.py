#!/usr/bin/env python3
"""
Prepare SAE decoder vectors and labels for SelfIE adapter training.

This script:
1. Fetches an SAE from SAELens given a release/id
2. Fetches auto-interpretability labels (Goodfire JSON or Neuronpedia S3 for LlamaScope)
3. Saves a .pt file with decoder vectors
4. Saves a .json file in the training format with train/val splits

Usage:
    # LlamaScope example (32k, layer 19, residual stream)
    python prepare_sae_training_data.py \\
        --release llama_scope_lxr_8x \\
        --sae-id l19r_8x \\
        --output-dir data/ \\
        --output-name llamascope_l19r_32k

    # Goodfire 70B example (auto-downloads labels from Neuronpedia)
    python prepare_sae_training_data.py \\
        --release goodfire-llama-3.3-70b-instruct \\
        --sae-id layer_50 \\
        --output-dir data/ \\
        --output-name goodfire_70b_l50

    # Goodfire 8B example (requires manual labels JSON - not on Neuronpedia)
    python prepare_sae_training_data.py \\
        --release goodfire-llama-3.1-8b-instruct \\
        --sae-id layer_19 \\
        --output-dir data/ \\
        --output-name goodfire_8b_l19 \\
        --labels-json path/to/goodfire_labels.json

    # With custom labels file (if you already downloaded)
    python prepare_sae_training_data.py \\
        --release llama_scope_lxr_8x \\
        --sae-id l19r_8x \\
        --output-dir data/ \\
        --output-name llamascope_l19r_32k \\
        --labels-jsonl path/to/labels.jsonl
"""

import argparse
import gzip
import io
import json
from pathlib import Path

import numpy as np
import requests
import torch

# SAELens release -> Neuronpedia S3 naming mapping
# For LlamaScope: (model_path, site, width) where site: res=residual, mlp=mlp_out, attn=attention_out
# For Goodfire: (model_path, neuronpedia_suffix) - full path prefix
SAELENS_TO_NEURONPEDIA = {
    # LlamaScope Residual stream SAEs (8x = 32k, 32x = 131k)
    "llama_scope_lxr_8x": ("llama3.1-8b", "llamascope-res", "32k"),
    "llama_scope_lxr_32x": ("llama3.1-8b", "llamascope-res", "131k"),
    # LlamaScope MLP output SAEs
    "llama_scope_lxm_8x": ("llama3.1-8b", "llamascope-mlp", "32k"),
    "llama_scope_lxm_32x": ("llama3.1-8b", "llamascope-mlp", "131k"),
    # LlamaScope Attention output SAEs
    "llama_scope_lxa_8x": ("llama3.1-8b", "llamascope-attn", "32k"),
    "llama_scope_lxa_32x": ("llama3.1-8b", "llamascope-attn", "131k"),
    # Goodfire SAEs (70B labels available on Neuronpedia; 8B labels not available)
    "goodfire-llama-3.3-70b-instruct": ("llama3.3-70b-it", "resid-post-gf", None),
}

# S3 bucket base URL for Neuronpedia explanations
NEURONPEDIA_S3_BASE = "https://neuronpedia-datasets.s3.amazonaws.com"


def parse_layer_from_sae_id(sae_id: str) -> int:
    """Extract layer number from SAE ID like 'l19r_8x' or 'layer_19'."""
    if sae_id.startswith("layer_"):
        # Goodfire format: layer_19
        return int(sae_id.split("_")[1])
    elif sae_id.startswith("l") and "_" in sae_id:
        # LlamaScope format: l19r_8x, l19m_32x, etc.
        layer_part = sae_id.split("_")[0]  # e.g., "l19r"
        # Remove 'l' prefix and any trailing letters (r, m, a, tc)
        layer_str = layer_part[1:].rstrip("ramtc")
        return int(layer_str)
    else:
        raise ValueError(f"Cannot parse layer from SAE ID: {sae_id}")


def get_neuronpedia_explanations_url(release: str, layer: int) -> str:
    """Get the S3 prefix URL for explanations based on release and layer."""
    if release not in SAELENS_TO_NEURONPEDIA:
        raise ValueError(
            f"Release '{release}' not supported for automatic label download. "
            f"Supported releases: {list(SAELENS_TO_NEURONPEDIA.keys())}"
        )

    mapping = SAELENS_TO_NEURONPEDIA[release]
    model_path, site_or_suffix, width = mapping

    if width is not None:
        # LlamaScope format: v1/{model}/{layer}-{site}-{width}/explanations/
        prefix = f"v1/{model_path}/{layer}-{site_or_suffix}-{width}/explanations"
    else:
        # Goodfire format: v1/{model}/{layer}-{suffix}/explanations/
        prefix = f"v1/{model_path}/{layer}-{site_or_suffix}/explanations"

    return f"{NEURONPEDIA_S3_BASE}/{prefix}"


def list_s3_batch_files(base_url: str) -> list[str]:
    """
    List batch files from S3 directory.

    Returns URLs of all batch-*.jsonl.gz files.
    """
    # Construct the S3 listing URL
    bucket = "neuronpedia-datasets"
    prefix = base_url.replace(NEURONPEDIA_S3_BASE + "/", "")

    list_url = f"https://{bucket}.s3.us-east-1.amazonaws.com/?prefix={prefix}/&max-keys=1000"

    print(f"📂 Listing S3 bucket: {list_url}")

    try:
        response = requests.get(list_url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to list S3 bucket: {e}")

    # Parse XML to find batch files
    import xml.etree.ElementTree as ET

    root = ET.fromstring(response.content)

    # Handle S3 XML namespace
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

    batch_files = []
    for contents in root.findall("s3:Contents", ns):
        key_elem = contents.find("s3:Key", ns)
        if key_elem is not None:
            key = key_elem.text
            if key and "batch-" in key and key.endswith(".jsonl.gz"):
                batch_files.append(f"{NEURONPEDIA_S3_BASE}/{key}")

    if not batch_files:
        raise RuntimeError(
            f"No batch files found at {base_url}/. "
            "Check that explanations exist for this SAE layer."
        )

    print(f"  Found {len(batch_files)} batch files")
    return sorted(batch_files)


def download_and_parse_jsonl_gz(url: str) -> list[dict]:
    """Download and parse a gzipped JSONL file from URL."""
    print(f"  Downloading: {url.split('/')[-1]}")

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download {url}: {e}")

    # Decompress and parse
    with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
        lines = f.read().decode("utf-8").strip().split("\n")

    entries = []
    for line in lines:
        if line.strip():
            entries.append(json.loads(line))

    return entries


def fetch_neuronpedia_labels(release: str, layer: int) -> dict[int, str]:
    """
    Fetch auto-interpretability labels from Neuronpedia S3 bucket.

    Returns:
        Dict mapping feature index -> label string
    """
    base_url = get_neuronpedia_explanations_url(release, layer)
    print(f"\n📥 Fetching labels from Neuronpedia:")
    print(f"  URL: {base_url}")

    batch_files = list_s3_batch_files(base_url)

    labels = {}
    for url in batch_files:
        entries = download_and_parse_jsonl_gz(url)
        for entry in entries:
            # Neuronpedia format has 'index' as string and 'description' as label
            idx = int(entry["index"])
            description = entry.get("description", "").strip()
            if description:
                # Remove leading space that Neuronpedia sometimes adds
                if description.startswith(" "):
                    description = description[1:]
                labels[idx] = description

    print(f"  ✓ Loaded {len(labels)} labels")
    return labels


def load_goodfire_labels_json(json_path: str) -> dict[int, str]:
    """
    Load Goodfire-format labels from a JSON file.

    Expected format (either at top level or as first element of array):
    {
        "1": "label for feature 1",
        "2": "label for feature 2",
        ...
    }
    """
    print(f"\n📥 Loading Goodfire labels from: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    # Handle array wrapper (like our training format)
    if isinstance(data, list) and len(data) > 0:
        data = data[0]
        if "vectors" in data:
            # Training format: extract from vectors list
            labels = {}
            for item in data["vectors"]:
                idx = item["index"]
                # Take first label if multiple
                if item.get("labels"):
                    labels[idx] = item["labels"][0]
            print(f"  ✓ Loaded {len(labels)} labels (training format)")
            return labels

    # Direct mapping format
    if isinstance(data, dict):
        labels = {int(k): v for k, v in data.items() if isinstance(v, str)}
        print(f"  ✓ Loaded {len(labels)} labels")
        return labels

    raise ValueError(f"Unrecognized JSON format in {json_path}")


def load_jsonl_labels(jsonl_path: str) -> dict[int, str]:
    """
    Load labels from a JSONL file (optionally gzipped).

    Each line should have 'index' and 'description' fields.
    """
    print(f"\n📥 Loading labels from: {jsonl_path}")

    labels = {}

    open_func = gzip.open if jsonl_path.endswith(".gz") else open
    mode = "rt" if jsonl_path.endswith(".gz") else "r"

    with open_func(jsonl_path, mode) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            idx = int(entry["index"])
            description = entry.get("description", "").strip()
            if description:
                if description.startswith(" "):
                    description = description[1:]
                labels[idx] = description

    print(f"  ✓ Loaded {len(labels)} labels")
    return labels


def create_train_val_split(
    num_features: int,
    labels: dict[int, str],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Create train/val split for labeled features.

    Returns:
        Tuple of (train_items, val_items) where each item has:
        {"index": int, "labels": [str], "split": "train"|"val"}
    """
    rng = np.random.RandomState(seed)

    # Get indices that have labels
    labeled_indices = sorted(labels.keys())
    print(f"\n📊 Creating train/val split:")
    print(f"  Total features: {num_features}")
    print(f"  Labeled features: {len(labeled_indices)}")

    # Shuffle and split
    shuffled = labeled_indices.copy()
    rng.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * val_fraction))
    val_indices = set(shuffled[:val_count])

    train_items = []
    val_items = []

    for idx in labeled_indices:
        label = labels[idx]
        item = {"index": idx, "labels": [label]}

        if idx in val_indices:
            item["split"] = "val"
            val_items.append(item)
        else:
            item["split"] = "train"
            train_items.append(item)

    print(f"  Train: {len(train_items)}")
    print(f"  Val: {len(val_items)}")

    return train_items, val_items


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SAE decoder vectors and labels for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--release",
        type=str,
        required=True,
        help="SAELens release identifier (e.g., 'llama_scope_lxr_8x', 'goodfire-llama-3.1-8b-instruct')",
    )
    parser.add_argument(
        "--sae-id",
        type=str,
        required=True,
        help="SAE identifier within the release (e.g., 'l19r_8x', 'layer_19')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save output files",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        required=True,
        help="Base name for output files (will create {name}.pt and {name}_labels.json)",
    )
    parser.add_argument(
        "--labels-json",
        type=str,
        default=None,
        help="Path to Goodfire-format labels JSON (if not fetching from Neuronpedia)",
    )
    parser.add_argument(
        "--labels-jsonl",
        type=str,
        default=None,
        help="Path to JSONL labels file (if not fetching from Neuronpedia)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of labeled features to use for validation (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load SAE on (default: cpu)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output file paths
    vectors_path = output_dir / f"{args.output_name}.pt"
    labels_path = output_dir / f"{args.output_name}_labels.json"

    print("=" * 70)
    print("SAE Training Data Preparation")
    print("=" * 70)

    # Step 1: Load SAE
    print(f"\n📦 Loading SAE from SAELens:")
    print(f"  Release: {args.release}")
    print(f"  SAE ID: {args.sae_id}")

    from sae_lens import SAE

    sae = SAE.from_pretrained(
        release=args.release,
        sae_id=args.sae_id,
        device=args.device,
    )

    num_features = sae.cfg.d_sae
    model_dim = sae.cfg.d_in
    print(f"  ✓ Loaded SAE: {num_features} features, dim={model_dim}")

    # Get decoder vectors
    decoder_vectors = sae.W_dec.detach().cpu()
    print(f"  Decoder shape: {decoder_vectors.shape}")

    # Step 2: Get labels
    layer = parse_layer_from_sae_id(args.sae_id)
    print(f"\n  Parsed layer: {layer}")

    if args.labels_json:
        labels = load_goodfire_labels_json(args.labels_json)
    elif args.labels_jsonl:
        labels = load_jsonl_labels(args.labels_jsonl)
    else:
        # Auto-fetch from Neuronpedia
        # Note: Goodfire 8B labels are not on Neuronpedia, but 70B labels are
        if args.release == "goodfire-llama-3.1-8b-instruct":
            raise ValueError(
                "Goodfire 8B labels must be provided via --labels-json. "
                "Automatic download is only supported for Goodfire 70B and LlamaScope SAEs."
            )
        labels = fetch_neuronpedia_labels(args.release, layer)

    # Step 3: Create train/val split
    train_items, val_items = create_train_val_split(
        num_features, labels, args.val_fraction, args.seed
    )

    # Step 4: Build metadata
    # Determine source and dataset name
    if "goodfire" in args.release.lower():
        source = "Goodfire"
        dataset_name = f"Goodfire/Llama-SAE-l{layer}"
    else:
        source = "Neuronpedia/LlamaScope"
        # Parse position from release name
        if "lxr" in args.release:
            pos = "res"
        elif "lxm" in args.release:
            pos = "mlp"
        elif "lxa" in args.release:
            pos = "attn"
        else:
            pos = "unknown"
        width = "32k" if "8x" in args.release else "131k"
        dataset_name = f"LlamaScope/l{layer}{pos[0]}_{width}"

    metadata = {
        "dataset_name": dataset_name,
        "filename": f"{args.output_name}.pt",
        "layer": layer,
        "source": source,
        "vector_type": "sae_decoder",
        "num_features": num_features,
        "model_dim": model_dim,
        "saelens_release": args.release,
        "saelens_sae_id": args.sae_id,
    }

    # Step 5: Save outputs
    print(f"\n💾 Saving outputs:")

    # Save decoder vectors
    print(f"  Vectors: {vectors_path}")
    torch.save(decoder_vectors, vectors_path)

    # Save labels JSON
    output_data = [{"metadata": metadata, "vectors": train_items + val_items}]

    print(f"  Labels: {labels_path}")
    with open(labels_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("✅ Done!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  • {vectors_path}")
    print(f"  • {labels_path}")
    print(f"\nTo use for training, reference the labels file in your config:")
    print(f'  labels_file: "{labels_path}"')
    print("\nStatistics:")
    print(f"  • Total features: {num_features}")
    print(f"  • Labeled features: {len(labels)}")
    print(f"  • Train: {len(train_items)}")
    print(f"  • Val: {len(val_items)}")


if __name__ == "__main__":
    main()
