#!/usr/bin/env python3
"""
Quick script to extract and save mean vectors for already-extracted Wikipedia dataset.
This only extracts the mean vectors without re-extracting all contrastive vectors.
"""

import json
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def main():
    # Configuration
    dataset_name = "keenanpepper/fifty-thousand-things"
    output_dir = Path("outputs/qwen_scaling")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    base_name = "qwen25_7b_instruct_fifty_thousand_things"
    device = "cuda"
    batch_size = 32
    
    # Load summary to get layer info
    summary_file = output_dir / f"{base_name}_summary.json"
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    layer_indices = summary['layer_indices']
    
    print("=" * 80)
    print("Extracting Mean Vectors Only")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Layers: {layer_indices}")
    print("=" * 80)
    print()
    
    # Load dataset
    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train")
    num_items = len(dataset)
    print(f"Loaded {num_items} items\n")
    
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.clean_up_tokenization_spaces = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"✓ Model loaded\n")
    
    # Extract vectors and compute running mean (much faster than concatenating!)
    print("Extracting raw vectors and computing means...")
    
    # Initialize running sums for each layer
    running_sums = {layer_idx: None for layer_idx in layer_indices}
    
    for i in tqdm(range(0, num_items, batch_size), desc="Extracting"):
        batch_end = min(i + batch_size, num_items)
        batch_data = dataset[i:batch_end]
        
        # Format prompts
        batch_prompts = batch_data["prompt"] if isinstance(batch_data["prompt"], list) else [batch_data["prompt"]]
        batch_formatted = []
        for prompt in batch_prompts:
            conversation = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            batch_formatted.append(formatted)
        
        # Tokenize
        tokens = tokenizer(
            batch_formatted,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True,
            )
            
            # Extract from each layer and accumulate sum
            for layer_idx in layer_indices:
                hidden_states = outputs.hidden_states[layer_idx + 1]
                last_token_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_dim]
                
                # Sum across batch dimension
                batch_sum = last_token_hidden.sum(dim=0).cpu()  # [hidden_dim]
                
                # Add to running sum
                if running_sums[layer_idx] is None:
                    running_sums[layer_idx] = batch_sum
                else:
                    running_sums[layer_idx] += batch_sum
    
    # Compute and save mean vectors
    print("\nSaving mean vectors...")
    for layer_idx in layer_indices:
        # Compute mean from running sum
        mean_vector = running_sums[layer_idx] / num_items  # [hidden_dim]
        mean_norm = torch.norm(mean_vector).item()
        
        # Save
        mean_file = output_dir / f"{base_name}_l{layer_idx}_mean.pt"
        torch.save(mean_vector, mean_file)
        print(f"  Layer {layer_idx}: norm={mean_norm:.2f} -> {mean_file.name}")
    
    print()
    print("=" * 80)
    print("✓ Complete! Mean vectors saved.")
    print("=" * 80)


if __name__ == "__main__":
    main()
