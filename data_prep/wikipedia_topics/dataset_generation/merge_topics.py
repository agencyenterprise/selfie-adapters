#!/usr/bin/env python3
"""Merge multiple generated topic files, deduplicating labels.

This script merges results from multiple generation runs, combining labels
while handling prompt differences. If multiple runs produced different prompts
for the same topic, it uses chosen_prompts.json if available (from
choose_best_prompts.py), otherwise falls back to the first prompt seen.

Usage:
    python merge_topics.py

Input: generated_topics_*.json files in outputs/ directory
Output: generated_topics_merged.json
"""

import json
from pathlib import Path
from collections import defaultdict
import sys


def load_topics(filepath):
    """Load topics from a generated JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return data["topics"]


def merge_topics(input_files, chosen_prompts=None):
    """
    Merge topics from multiple files.
    
    Args:
        input_files: List of file paths to merge
        chosen_prompts: Optional dict mapping original_title to chosen prompt
    
    Returns:
        - merged_topics: dict mapping original_title to merged topic data
        - prompt_differences: list of (title, prompts) where prompts differ
        - prompt_differences_resolved: number of prompt differences resolved by chosen_prompts
    """
    # Group topics by original_title
    topics_by_title = defaultdict(list)
    
    for filepath in input_files:
        print(f"Loading {filepath}...")
        topics = load_topics(filepath)
        print(f"  Loaded {len(topics)} topics")
        
        for topic in topics:
            topics_by_title[topic["original_title"]].append(topic)
    
    print(f"\nTotal unique titles: {len(topics_by_title)}")
    
    # Merge labels and check for prompt differences
    merged_topics = {}
    prompt_differences = []
    prompt_differences_resolved = 0
    
    for title, topic_list in topics_by_title.items():
        # Collect all prompts and labels
        prompts = [t["prompt"] for t in topic_list]
        all_labels = []
        for t in topic_list:
            all_labels.extend(t["labels"])
        
        # Check if prompts differ
        unique_prompts = list(set(prompts))
        has_prompt_difference = len(unique_prompts) > 1
        
        if has_prompt_difference:
            # Check if we have a chosen prompt for this title
            if chosen_prompts and title in chosen_prompts:
                selected_prompt = chosen_prompts[title]
                prompt_differences_resolved += 1
            else:
                # Use the first prompt as fallback
                selected_prompt = prompts[0]
                prompt_differences.append((title, unique_prompts))
        else:
            selected_prompt = prompts[0]
        
        merged_topics[title] = {
            "original_title": title,
            "prompt": selected_prompt,
            "labels": list(dict.fromkeys(all_labels))  # Deduplicate while preserving order
        }
    
    return merged_topics, prompt_differences, prompt_differences_resolved


def main():
    # Find all generated_topics_*.json files
    output_dir = Path(__file__).parent / "outputs"
    input_files = sorted(output_dir.glob("generated_topics_*.json"))
    
    # Exclude the merged file itself
    input_files = [f for f in input_files if "merged" not in f.name and "filtered" not in f.name]
    
    if not input_files:
        print("Error: No generated_topics_*.json files found in outputs/!")
        print("Run generate_all.py and check_batch.py first.")
        sys.exit(1)
    
    print(f"Found {len(input_files)} input files:")
    for f in input_files:
        print(f"  - {f.name}")
    print()
    
    # Load valid titles from vital_articles_level5.json
    data_dir = Path(__file__).parent.parent
    vital_articles_file = data_dir / "vital_articles_level5.json"
    print(f"Loading valid titles from {vital_articles_file.name}...")
    with open(vital_articles_file) as f:
        vital_articles_data = json.load(f)
    valid_titles = set(vital_articles_data["titles"])
    print(f"Loaded {len(valid_titles)} valid titles\n")
    
    # Load chosen prompts if available
    chosen_prompts_file = output_dir / "chosen_prompts.json"
    chosen_prompts = None
    if chosen_prompts_file.exists():
        print(f"Loading chosen prompts from {chosen_prompts_file.name}...")
        with open(chosen_prompts_file) as f:
            chosen_prompts = json.load(f)
        print(f"Loaded {len(chosen_prompts)} chosen prompts\n")
    
    # Merge the topics
    merged_topics, prompt_differences, prompt_differences_resolved = merge_topics(input_files, chosen_prompts)
    
    # Filter to only valid titles
    before_filter = len(merged_topics)
    merged_topics = {
        title: topic 
        for title, topic in merged_topics.items() 
        if title in valid_titles
    }
    after_filter = len(merged_topics)
    filtered_out = before_filter - after_filter
    
    if filtered_out > 0:
        print(f"\n⚠️  Filtered out {filtered_out} topics not in vital_articles_level5.json")
    else:
        print("\n✓ All topics are in vital_articles_level5.json")
    
    # Report prompt differences
    if chosen_prompts:
        print(f"✓ Resolved {prompt_differences_resolved} prompt differences using chosen_prompts.json")
    
    if prompt_differences:
        print("\n" + "="*80)
        print(f"WARNING: Found {len(prompt_differences)} titles with differing prompts (unresolved):")
        print("="*80)
        for title, prompts in prompt_differences[:10]:  # Show first 10
            print(f"\nTitle: {title}")
            for i, prompt in enumerate(prompts, 1):
                print(f"  Prompt {i}: {prompt}")
        if len(prompt_differences) > 10:
            print(f"\n... and {len(prompt_differences) - 10} more")
        print("="*80)
        print("\nRun choose_best_prompts.py to resolve these prompt differences.")
    else:
        print("\n✓ All prompts match (or resolved via chosen_prompts.json)")
    
    # Calculate statistics
    total_topics = len(merged_topics)
    total_labels = sum(len(t["labels"]) for t in merged_topics.values())
    avg_labels = total_labels / total_topics if total_topics > 0 else 0
    
    print("\nMerge Statistics:")
    print(f"  Total unique topics: {total_topics}")
    print(f"  Total labels (after dedup): {total_labels}")
    print(f"  Average labels per topic: {avg_labels:.2f}")
    
    # Create output with metadata
    output = {
        "topics": list(merged_topics.values()),
        "metadata": {
            "total": total_topics,
            "total_labels": total_labels,
            "average_labels_per_topic": avg_labels,
            "num_input_files": len(input_files),
            "input_files": [f.name for f in input_files],
            "filtered_out": filtered_out,
            "valid_titles_source": "vital_articles_level5.json",
            "prompt_differences_resolved": prompt_differences_resolved,
            "prompt_differences_unresolved": len(prompt_differences),
            "used_chosen_prompts": chosen_prompts is not None
        }
    }
    
    # Save merged output
    output_file = output_dir / "generated_topics_merged.json"
    print(f"\nSaving merged topics to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Saved {total_topics} merged topics to {output_file.name}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
