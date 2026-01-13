#!/usr/bin/env python3
"""
Extract residual stream vectors from Wikipedia article prompts.

For each Wikipedia article title:
1. Create prompt: "Tell me about {title}."
2. Extract residual stream vector from layer 19 at the last token before the assistant response
3. Compute mean of all vectors
4. Create contrastive vectors (vector - mean)
5. Save as (vector, label) dataset
"""

import json
import torch
import numpy as np
from pathlib import Path
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

    return model, tokenizer


def create_prompt(title: str) -> str:
    """Create a prompt for a Wikipedia article title."""
    return f"Tell me about {title}."


def format_conversation(prompt: str, tokenizer) -> str:
    """
    Format a single-turn conversation using the chat template.

    Args:
        prompt: User prompt
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


def get_residual_stream_at_layer_batched(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """
    Run batched forward pass and extract residual stream at specified layer.

    Args:
        model: The language model
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        layer_idx: Which layer to extract from (0-indexed)

    Returns:
        Residual stream activations at the last token position [batch_size, hidden_dim]
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # outputs.hidden_states[layer_idx + 1] gives us the output after layer_idx
        hidden_states = outputs.hidden_states[layer_idx + 1]

        # With LEFT padding, the last real token is always at position -1
        last_token_hidden_states = hidden_states[:, -1, :]  # [batch_size, hidden_dim]

        return last_token_hidden_states


def extract_all_vectors(
    titles: List[str],
    model,
    tokenizer,
    layer_idx: int,
    device: str = "cuda",
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Extract residual stream vectors for all titles.

    Returns:
        Tensor of shape [num_titles, hidden_dim]
    """
    all_vectors = []
    num_titles = len(titles)

    print(f"Extracting vectors for {num_titles} titles...")

    for i in tqdm(range(0, num_titles, batch_size), desc="Extracting vectors"):
        batch_end = min(i + batch_size, num_titles)
        batch_titles = titles[i:batch_end]

        # Create prompts and format as conversations
        batch_prompts = [create_prompt(title) for title in batch_titles]
        batch_formatted = [format_conversation(prompt, tokenizer) for prompt in batch_prompts]

        # Tokenize batch
        tokens = tokenizer(
            batch_formatted,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)

        # Get activations
        vectors = get_residual_stream_at_layer_batched(
            model,
            tokens.input_ids,
            tokens.attention_mask,
            layer_idx,
        )

        all_vectors.append(vectors.cpu())

    # Concatenate all batches
    all_vectors_tensor = torch.cat(all_vectors, dim=0)  # [num_titles, hidden_dim]

    return all_vectors_tensor


def main(
    titles_file: str = "vital_articles_level5.json",
    output_vectors_file: str = "outputs/wikipedia_vectors_l19.pt",
    output_metadata_file: str = "outputs/wikipedia_metadata_l19.json",
    output_dataset_file: str = "outputs/wikipedia_contrastive_dataset.pt",
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    layer_idx: int = 19,
    device: str = "cuda",
    batch_size: int = 32,
):
    """Main function to extract vectors and create contrastive dataset."""

    print("=" * 80)
    print("Extracting Wikipedia Article Vectors")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Layer: {layer_idx}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Input titles: {titles_file}")
    print(f"Output vectors: {output_vectors_file}")
    print(f"Output metadata: {output_metadata_file}")
    print(f"Output dataset: {output_dataset_file}")
    print("=" * 80)
    print()

    # Load titles
    print(f"Loading titles from {titles_file}...")
    with open(titles_file, "r") as f:
        titles_data = json.load(f)

    titles = titles_data["titles"]
    num_titles = len(titles)
    print(f"Loaded {num_titles} titles")
    print()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    hidden_dim = model.config.hidden_size

    # Extract all vectors
    all_vectors = extract_all_vectors(
        titles,
        model,
        tokenizer,
        layer_idx,
        device=device,
        batch_size=batch_size,
    )

    print(f"\nExtracted vectors shape: {all_vectors.shape}")

    # Compute mean vector
    print("Computing mean vector...")
    mean_vector = all_vectors.mean(dim=0)  # [hidden_dim]
    print(f"Mean vector shape: {mean_vector.shape}")

    # Create contrastive vectors (vector - mean)
    print("Creating contrastive vectors...")
    contrastive_vectors = all_vectors - mean_vector.unsqueeze(0)  # [num_titles, hidden_dim]
    print(f"Contrastive vectors shape: {contrastive_vectors.shape}")

    # Save raw vectors
    print(f"\nSaving raw vectors to {output_vectors_file}...")
    torch.save(all_vectors, output_vectors_file)

    # Save metadata
    print(f"Saving metadata to {output_metadata_file}...")
    metadata = {
        "model_name": model_name,
        "layer_idx": layer_idx,
        "num_titles": num_titles,
        "hidden_dim": hidden_dim,
        "mean_vector_norm": float(torch.norm(mean_vector).item()),
        "titles": titles,
    }
    with open(output_metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    # Create dataset with (vector, label) pairs
    print(f"Creating and saving dataset to {output_dataset_file}...")
    dataset = {
        "vectors": contrastive_vectors,  # [num_titles, hidden_dim]
        "labels": titles,  # List of strings
        "mean_vector": mean_vector,  # [hidden_dim]
        "metadata": {
            "model_name": model_name,
            "layer_idx": layer_idx,
            "num_samples": num_titles,
            "hidden_dim": hidden_dim,
        }
    }
    torch.save(dataset, output_dataset_file)

    print()
    print("=" * 80)
    print("✓ Complete!")
    print(f"✓ Saved {num_titles} raw vectors to {output_vectors_file}")
    print(f"✓ Saved metadata to {output_metadata_file}")
    print(f"✓ Saved contrastive dataset to {output_dataset_file}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract Wikipedia article vectors")
    parser.add_argument("--titles-file", type=str,
                        default="vital_articles_level5.json",
                        help="Input JSON file with Wikipedia titles")
    parser.add_argument("--output-vectors", type=str,
                        default="outputs/wikipedia_vectors_l19.pt",
                        help="Output file for raw vectors (PyTorch tensor)")
    parser.add_argument("--output-metadata", type=str,
                        default="outputs/wikipedia_metadata_l19.json",
                        help="Output file for metadata (JSON)")
    parser.add_argument("--output-dataset", type=str,
                        default="outputs/wikipedia_contrastive_dataset.pt",
                        help="Output file for contrastive dataset")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="Model name or path")
    parser.add_argument("--layer", type=int, default=19,
                        help="Layer index to extract activations from")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda or cpu)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for forward passes (default: 32)")

    args = parser.parse_args()

    main(
        titles_file=args.titles_file,
        output_vectors_file=args.output_vectors,
        output_metadata_file=args.output_metadata,
        output_dataset_file=args.output_dataset,
        model_name=args.model,
        layer_idx=args.layer,
        device=args.device,
        batch_size=args.batch_size,
    )
