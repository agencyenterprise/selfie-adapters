#!/usr/bin/env python3
"""Generate batch requests to choose best prompts for entries with multiple prompts.

When multiple generation runs produce different prompts for the same topic,
this script uses Claude to select the best prompt for each.

Usage:
    python choose_best_prompts.py

After running, use check_chosen_prompts.py to retrieve results.
Then re-run merge_topics.py to use the chosen prompts.
"""

import json
import time
from pathlib import Path
from collections import defaultdict
from anthropic import Anthropic


def main():
    # Initialize Anthropic client
    client = Anthropic()
    
    output_dir = Path(__file__).parent / "outputs"
    prompts_dir = Path(__file__).parent / "prompts"
    
    # Load the prompt template
    with open(prompts_dir / "choose_prompt_prompt.txt") as f:
        prompt_template = f.read()
    
    print("Loading generated topic files to find entries with multiple prompts...")
    
    # Load all generated topic files and find entries with differing prompts
    input_files = sorted(output_dir.glob("generated_topics_*.json"))
    input_files = [f for f in input_files if "merged" not in f.name and "filtered" not in f.name]
    
    topics_by_title = defaultdict(list)
    
    for filepath in input_files:
        print(f"  Loading {filepath.name}...")
        with open(filepath) as f:
            data = json.load(f)
        for topic in data["topics"]:
            topics_by_title[topic["original_title"]].append(topic)
    
    # Find entries with differing prompts
    entries_with_multiple_prompts = []
    
    for title, topic_list in topics_by_title.items():
        prompts = [t["prompt"] for t in topic_list]
        unique_prompts = list(set(prompts))
        
        if len(unique_prompts) > 1:
            # Collect all labels
            all_labels = []
            for t in topic_list:
                all_labels.extend(t["labels"])
            # Deduplicate labels
            unique_labels = list(dict.fromkeys(all_labels))
            
            entries_with_multiple_prompts.append({
                "original_title": title,
                "prompt_options": unique_prompts,
                "labels": unique_labels
            })
    
    print(f"\nFound {len(entries_with_multiple_prompts)} entries with multiple prompts")
    
    if len(entries_with_multiple_prompts) == 0:
        print("No entries need prompt selection. Exiting.")
        exit(0)
    
    # Create batch requests
    print("\nPreparing batch requests...")
    batch_requests = []
    
    for i, entry in enumerate(entries_with_multiple_prompts):
        # Format the prompt
        prompt_options_text = "\n".join(f"{j+1}. {p}" for j, p in enumerate(entry["prompt_options"]))
        labels_text = "\n".join(f"- {label}" for label in entry["labels"])
        
        full_prompt = prompt_template.format(
            original_title=entry["original_title"],
            prompt_options=prompt_options_text,
            labels=labels_text
        )
        
        # Create a safe custom_id
        custom_id = f"choose_prompt_{i:05d}"
        
        batch_requests.append({
            "custom_id": custom_id,
            "params": {
                "model": "claude-sonnet-4-5",
                "max_tokens": 200,  # Short response, just the prompt
                "messages": [
                    {"role": "user", "content": full_prompt}
                ]
            }
        })
    
    print(f"Created {len(batch_requests)} batch requests")
    
    # Save the entries for later reference
    entries_file = output_dir / "entries_needing_prompt_selection.json"
    with open(entries_file, "w") as f:
        json.dump(entries_with_multiple_prompts, f, indent=2)
    print(f"Saved entries to {entries_file.name}")
    
    # Save batch requests to file
    batch_file = output_dir / "choose_prompts_batch_requests.jsonl"
    print(f"\nWriting batch requests to {batch_file.name}...")
    with open(batch_file, "w") as f:
        for request in batch_requests:
            f.write(json.dumps(request) + "\n")
    
    print(f"Wrote {len(batch_requests)} batch requests")
    
    # Submit the batch
    print("\nSubmitting batch to Anthropic...")
    message_batch = client.messages.batches.create(
        requests=batch_requests
    )
    
    batch_id = message_batch.id
    print(f"Batch submitted! ID: {batch_id}")
    print(f"Status: {message_batch.processing_status}")
    
    # Save batch ID and metadata
    batch_info_file = output_dir / "choose_prompts_batch_info.json"
    with open(batch_info_file, "w") as f:
        json.dump({
            "batch_id": batch_id,
            "submitted_at": time.time(),
            "num_requests": len(batch_requests),
            "total_entries": len(entries_with_multiple_prompts)
        }, f, indent=2)
    
    print(f"\nBatch info saved to {batch_info_file.name}")
    print("\nRun check_chosen_prompts.py to check status and retrieve results.")
    print(f"Expected processing time: ~5-10 minutes for {len(batch_requests)} requests")


if __name__ == "__main__":
    main()
