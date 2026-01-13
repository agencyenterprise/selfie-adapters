#!/usr/bin/env python3
"""Generate topic dataset for all Wikipedia titles using Anthropic Batch API.

This is the first step in the dataset generation pipeline. It:
1. Loads Wikipedia article titles from vital_articles_level5.json
2. Groups titles into batches (20 per request)
3. Submits batches to Anthropic Batch API for generation
4. Saves batch metadata for later retrieval

Usage:
    python generate_all.py

After running, use check_batch.py to monitor status and retrieve results.
For multiple runs, rename the output files before running again.
"""

import json
import time
from pathlib import Path
from anthropic import Anthropic


def main():
    # Initialize Anthropic client (uses ANTHROPIC_API_KEY environment variable)
    client = Anthropic()
    
    # Load the data
    data_dir = Path(__file__).parent.parent
    with open(data_dir / "vital_articles_level5.json") as f:
        data = json.load(f)
    
    prompts_dir = Path(__file__).parent / "prompts"
    with open(prompts_dir / "generation_prompt.txt") as f:
        prompt_template = f.read()
    
    print(f"Preparing batch requests for {len(data['titles'])} Wikipedia titles...")
    
    # Prepare batch requests
    # Group titles into batches for efficiency (e.g., 20 titles per request)
    TITLES_PER_REQUEST = 20
    batch_requests = []
    all_titles = data["titles"]
    
    for batch_idx in range(0, len(all_titles), TITLES_PER_REQUEST):
        batch_titles = all_titles[batch_idx:batch_idx + TITLES_PER_REQUEST]
        full_prompt = prompt_template + "\n".join(batch_titles)
        
        # Create a safe custom_id (only alphanumeric, underscore, hyphen, max 64 chars)
        # Use format: batch_INDEX where INDEX is zero-padded to 5 digits
        custom_id = f"batch_{batch_idx//TITLES_PER_REQUEST:05d}"
        
        batch_requests.append({
            "custom_id": custom_id,
            "params": {
                "model": "claude-sonnet-4-5",
                "max_tokens": 4000,  # Increased for multiple topics
                "messages": [
                    {"role": "user", "content": full_prompt}
                ]
            }
        })
    
    print(f"Grouped into {len(batch_requests)} batch requests ({TITLES_PER_REQUEST} titles each)")
    
    # Save batch requests to file (for reference)
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    batch_file = output_dir / "batch_requests.jsonl"
    print(f"Writing batch requests to {batch_file}...")
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
    
    # Save batch ID and metadata for later retrieval
    batch_info_file = output_dir / "batch_info.json"
    with open(batch_info_file, "w") as f:
        json.dump({
            "batch_id": batch_id,
            "submitted_at": time.time(),
            "num_requests": len(batch_requests),
            "titles_per_request": TITLES_PER_REQUEST,
            "total_titles": len(data["titles"])
        }, f, indent=2)
    
    print(f"\nBatch info saved to {batch_info_file}")
    print("\nThe batch is now processing. Run check_batch.py to check status and retrieve results.")
    print(f"Expected processing time: ~10-30 minutes for {len(batch_requests)} requests")


if __name__ == "__main__":
    main()
