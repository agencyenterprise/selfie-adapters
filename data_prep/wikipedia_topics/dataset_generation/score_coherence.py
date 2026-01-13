#!/usr/bin/env python3
"""Generate batch requests to score label coherence for all merged topics.

This step evaluates how coherently all labels for each topic refer to the
same unique entity/concept. Topics with low coherence scores will be filtered out.

Usage:
    python score_coherence.py

After running, use check_coherence_scores.py to retrieve results.
Then use filter_by_coherence.py to filter the dataset.
"""

import json
import time
from pathlib import Path
from anthropic import Anthropic


def main():
    # Initialize Anthropic client
    client = Anthropic()
    
    output_dir = Path(__file__).parent / "outputs"
    prompts_dir = Path(__file__).parent / "prompts"
    
    # Load the prompt template
    with open(prompts_dir / "coherence_scoring_prompt.txt") as f:
        prompt_template = f.read()
    
    # Load merged topics
    merged_file = output_dir / "generated_topics_merged.json"
    if not merged_file.exists():
        print("Error: generated_topics_merged.json not found!")
        print("Run merge_topics.py first.")
        exit(1)
    
    print(f"Loading merged topics from {merged_file.name}...")
    with open(merged_file) as f:
        merged_data = json.load(f)
    
    topics = merged_data["topics"]
    print(f"Loaded {len(topics)} topics")
    
    # Create batch requests
    print("\nPreparing batch requests...")
    batch_requests = []
    
    for i, topic in enumerate(topics):
        # Format the labels
        labels_text = "\n".join(f"- {label}" for label in topic["labels"])
        
        full_prompt = prompt_template.format(
            original_title=topic["original_title"],
            prompt=topic["prompt"],
            labels=labels_text
        )
        
        # Create a safe custom_id
        custom_id = f"coherence_{i:05d}"
        
        batch_requests.append({
            "custom_id": custom_id,
            "params": {
                "model": "claude-sonnet-4-5",
                "max_tokens": 300,  # Room for reasoning + score
                "messages": [
                    {"role": "user", "content": full_prompt}
                ]
            }
        })
    
    print(f"Created {len(batch_requests)} batch requests")
    
    # Save batch requests to file
    batch_file = output_dir / "coherence_batch_requests.jsonl"
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
    batch_info_file = output_dir / "coherence_batch_info.json"
    with open(batch_info_file, "w") as f:
        json.dump({
            "batch_id": batch_id,
            "submitted_at": time.time(),
            "num_requests": len(batch_requests),
            "total_topics": len(topics)
        }, f, indent=2)
    
    print(f"\nBatch info saved to {batch_info_file.name}")
    print("\nRun check_coherence_scores.py to check status and retrieve results.")
    print(f"Expected processing time: ~30-60 minutes for {len(batch_requests)} requests")


if __name__ == "__main__":
    main()
