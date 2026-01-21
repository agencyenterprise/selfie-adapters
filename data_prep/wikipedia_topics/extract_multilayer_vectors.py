#!/usr/bin/env python3
"""
Extract residual stream vectors from multiple layers for scaling law experiments.

For each item in the dataset:
1. Use the pre-formatted prompt from the dataset
2. Extract residual stream vectors from multiple layers (middle 50% of model)
3. For each layer, save separate .pt file with contrastive vectors
4. Create unified JSON file in training format with all layers
"""

import json
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
from tqdm import tqdm


def load_model_and_tokenizer(model_name: str, device: str = "cuda", dtype=torch.bfloat16):
    """Load the model and tokenizer."""
    print(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.clean_up_tokenization_spaces = False

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Force LEFT padding
    tokenizer.padding_side = "left"

    print(f"Model loaded on {device} with dtype {dtype}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Number of layers: {model.config.num_hidden_layers}")

    return model, tokenizer


def format_conversation(prompt: str, tokenizer) -> str:
    """
    Format a single-turn conversation using the chat template.

    Args:
        prompt: User prompt (already formatted)
        tokenizer: The tokenizer with chat template

    Returns:
        Formatted conversation string ready for tokenization
    """
    conversation = [
        {"role": "user", "content": prompt}
    ]

    # Apply chat template with generation prompt to get the position right before
    # the assistant starts responding
    formatted = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    return formatted


def verify_chat_template(tokenizer, sample_prompt: str = "Tell me about Python."):
    """
    Verify the chat template formatting and show what the final tokens look like.
    """
    print("\n" + "=" * 80)
    print("CHAT TEMPLATE VERIFICATION")
    print("=" * 80)
    
    formatted = format_conversation(sample_prompt, tokenizer)
    print(f"\nSample prompt: {sample_prompt}")
    print(f"\nFormatted conversation:\n{repr(formatted)}")
    
    tokens = tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
    print(f"\nTokenized length: {tokens.input_ids.shape[1]}")
    print(f"Last 5 tokens: {tokens.input_ids[0, -5:].tolist()}")
    print(f"Last 5 decoded: {[tokenizer.decode([t]) for t in tokens.input_ids[0, -5:]]}")
    
    print("\n" + "=" * 80)
    print()


def get_residual_stream_at_layers_batched(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_indices: List[int],
) -> Dict[int, torch.Tensor]:
    """
    Run batched forward pass and extract residual stream at specified layers.

    Args:
        model: The language model
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        layer_indices: Which layers to extract from (0-indexed)

    Returns:
        Dict mapping layer_idx -> residual stream activations [batch_size, hidden_dim]
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        layer_activations = {}
        for layer_idx in layer_indices:
            # outputs.hidden_states[layer_idx + 1] gives us the output after layer_idx
            hidden_states = outputs.hidden_states[layer_idx + 1]
            
            # With LEFT padding, the last real token is always at position -1
            last_token_hidden_states = hidden_states[:, -1, :]  # [batch_size, hidden_dim]
            
            layer_activations[layer_idx] = last_token_hidden_states

        return layer_activations


def extract_all_vectors_by_layer(
    dataset,
    model,
    tokenizer,
    layer_indices: List[int],
    device: str = "cuda",
    batch_size: int = 32,
) -> Dict[int, torch.Tensor]:
    """
    Extract residual stream vectors from multiple layers for all items.

    Returns:
        Dict mapping layer_idx -> tensor of shape [num_items, hidden_dim]
    """
    num_items = len(dataset)
    hidden_dim = model.config.hidden_size
    
    # Pre-allocate tensors for each layer (much faster than list + concat!)
    vectors_by_layer = {
        layer_idx: torch.zeros(num_items, hidden_dim, dtype=torch.float32)
        for layer_idx in layer_indices
    }

    print(f"Extracting vectors for {num_items} items across {len(layer_indices)} layers...")
    print(f"Layer indices: {layer_indices}")
    print(f"Pre-allocated tensors: [{num_items}, {hidden_dim}] per layer")

    for i in tqdm(range(0, num_items, batch_size), desc="Extracting vectors"):
        batch_end = min(i + batch_size, num_items)
        batch_data = dataset[i:batch_end]

        # Use pre-formatted prompts from the dataset
        batch_prompts = batch_data["prompt"] if isinstance(batch_data["prompt"], list) else [batch_data["prompt"]]
        batch_formatted = [format_conversation(prompt, tokenizer) for prompt in batch_prompts]

        # Tokenize batch
        tokens = tokenizer(
            batch_formatted,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        
        # For device_map="auto", put inputs on the device of the first model parameter
        if device == "auto":
            first_device = next(model.parameters()).device
            tokens = tokens.to(first_device)
        else:
            tokens = tokens.to(device)

        # Get activations from all layers
        layer_activations = get_residual_stream_at_layers_batched(
            model,
            tokens.input_ids,
            tokens.attention_mask,
            layer_indices,
        )

        # Store vectors for each layer directly in pre-allocated tensor
        for layer_idx in layer_indices:
            vectors_by_layer[layer_idx][i:batch_end] = layer_activations[layer_idx].cpu()

    # Print final shapes
    for layer_idx in layer_indices:
        print(f"  Layer {layer_idx}: {vectors_by_layer[layer_idx].shape}")

    return vectors_by_layer


def calculate_layer_range(num_layers: int) -> List[int]:
    """
    Calculate the middle 50% (25%-75%) of layers.
    
    Args:
        num_layers: Total number of layers in the model
        
    Returns:
        List of layer indices to extract from
    """
    start_layer = int(num_layers * 0.25)
    end_layer = int(num_layers * 0.75)
    
    layer_range = list(range(start_layer, end_layer))
    
    print(f"\nModel has {num_layers} layers (0-{num_layers-1})")
    print(f"Extracting from middle 50%: layers {start_layer}-{end_layer-1} ({len(layer_range)} layers)")
    
    return layer_range


def extract_splits_from_dataset(dataset) -> List[str]:
    """
    Extract train/val split assignments from the dataset's existing split field.
    
    Returns:
        List of "train" or "val" for each item index
    """
    splits = [item["split"] for item in dataset]
    
    num_train = sum(1 for s in splits if s == "train")
    num_val = sum(1 for s in splits if s == "val")
    print(f"\nUsing existing dataset splits: {num_train} train, {num_val} val")
    
    return splits


def main(
    dataset_name: str = "keenanpepper/fifty-thousand-things",
    output_dir: str = "outputs/qwen_scaling",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    device: str = "cuda",
    batch_size: int = 32,
    split: str = "train",
):
    """Main function to extract multi-layer vectors and create training-format dataset."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output filenames based on model name
    model_short_name = model_name.split("/")[-1].replace("-", "_").lower().replace(".", "")
    base_name = f"{model_short_name}_fifty_thousand_things"
    output_labels_file = output_dir / f"{base_name}_labels.json"

    print("=" * 80)
    print("Extracting Multi-Layer Wikipedia Article Vectors")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Dataset split: {split}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    print()

    # Load dataset
    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, split=split)
    num_items = len(dataset)
    print(f"Loaded {num_items} items")
    print()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    hidden_dim = model.config.hidden_size
    num_model_layers = model.config.num_hidden_layers

    # Verify chat template
    verify_chat_template(tokenizer, dataset[0]["prompt"])

    # Calculate which layers to extract from (middle 50%)
    layer_indices = calculate_layer_range(num_model_layers)
    num_layers = len(layer_indices)

    # Extract all vectors, organized by layer
    vectors_by_layer = extract_all_vectors_by_layer(
        dataset,
        model,
        tokenizer,
        layer_indices,
        device=device,
        batch_size=batch_size,
    )

    # Compute per-layer mean vectors
    print("\nComputing per-layer mean vectors...")
    mean_vectors_by_layer = {}
    for layer_idx in layer_indices:
        mean_vectors_by_layer[layer_idx] = vectors_by_layer[layer_idx].mean(dim=0)  # [hidden_dim]
        print(f"  Layer {layer_idx}: mean norm = {torch.norm(mean_vectors_by_layer[layer_idx]).item():.4f}")

    # Create contrastive vectors for each layer (subtract per-layer mean)
    print("\nCreating contrastive vectors (subtracting per-layer means)...")
    contrastive_vectors_by_layer = {}
    for layer_idx in layer_indices:
        contrastive_vectors_by_layer[layer_idx] = vectors_by_layer[layer_idx] - mean_vectors_by_layer[layer_idx].unsqueeze(0)

    # Extract train/val split from dataset's existing split field
    splits = extract_splits_from_dataset(dataset)

    # Save .pt files (one per layer) and build JSON structure
    print(f"\nSaving .pt files and creating JSON metadata...")
    
    json_sections = []
    
    for layer_idx in layer_indices:
        # Save contrastive vectors .pt file for this layer
        layer_vectors_file = output_dir / f"{base_name}_l{layer_idx}.pt"
        torch.save(contrastive_vectors_by_layer[layer_idx], layer_vectors_file)
        print(f"  Saved layer {layer_idx} contrastive vectors: {layer_vectors_file.name}")
        
        # Save mean vector for this layer (needed for creating contrastive vectors from new topics)
        mean_vector_file = output_dir / f"{base_name}_l{layer_idx}_mean.pt"
        torch.save(mean_vectors_by_layer[layer_idx], mean_vector_file)
        print(f"  Saved layer {layer_idx} mean vector: {mean_vector_file.name}")
        
        # Build vectors list for this layer
        vectors_list = []
        for i in range(num_items):
            vectors_list.append({
                "index": i,
                "labels": dataset[i]["labels"],
                "split": splits[i],
            })
        
        # Create section for this layer
        section = {
            "metadata": {
                "dataset_name": "fifty_thousand_things",
                "filename": layer_vectors_file.name,  # Relative path
                "layer": layer_idx,
                "source": dataset_name,
                "vector_type": "contrastive",
                "model_name": model_name,
                "hidden_dim": hidden_dim,
                "num_vectors": num_items,
            },
            "vectors": vectors_list
        }
        json_sections.append(section)

    # Save JSON file
    print(f"\nSaving labels JSON to {output_labels_file}...")
    with open(output_labels_file, "w") as f:
        json.dump(json_sections, f, indent=2)

    # Save summary metadata
    summary_file = output_dir / f"{base_name}_summary.json"
    summary = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "num_items": num_items,
        "num_layers": num_layers,
        "layer_indices": layer_indices,
        "hidden_dim": hidden_dim,
        "mean_vector_norms_by_layer": {
            layer_idx: float(torch.norm(mean_vectors_by_layer[layer_idx]).item())
            for layer_idx in layer_indices
        },
        "train_count": sum(1 for s in splits if s == "train"),
        "val_count": sum(1 for s in splits if s == "val"),
        "output_files": {
            "labels": output_labels_file.name,
            "vectors_by_layer": [f"{base_name}_l{layer_idx}.pt" for layer_idx in layer_indices],
            "mean_vectors_by_layer": [f"{base_name}_l{layer_idx}_mean.pt" for layer_idx in layer_indices],
        }
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 80)
    print("✓ Complete!")
    print(f"✓ Saved {num_layers} contrastive vector .pt files (one per layer)")
    print(f"✓ Saved {num_layers} mean vector .pt files (one per layer)")
    print(f"✓ Saved labels JSON with {len(json_sections)} sections: {output_labels_file}")
    print(f"✓ Saved summary: {summary_file}")
    print(f"✓ Total: {num_items} items × {num_layers} layers = {num_items * num_layers} vectors")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract multi-layer Wikipedia article vectors")
    parser.add_argument("--dataset", type=str,
                        default="keenanpepper/fifty-thousand-things",
                        help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to use")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/qwen_scaling",
                        help="Output directory for all files")
    parser.add_argument("--model", type=str, 
                        default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name or path")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda or cpu)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for forward passes (default: 32)")

    args = parser.parse_args()

    main(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        split=args.split,
    )
