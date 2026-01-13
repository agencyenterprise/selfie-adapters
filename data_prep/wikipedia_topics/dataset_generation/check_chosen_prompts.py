#!/usr/bin/env python3
"""Check batch status and retrieve chosen prompts.

Usage:
    python check_chosen_prompts.py

After results are retrieved, re-run merge_topics.py to use the chosen prompts.
"""

import json
from pathlib import Path
from anthropic import Anthropic


def main():
    # Initialize Anthropic client
    client = Anthropic()
    
    output_dir = Path(__file__).parent / "outputs"
    
    # Load batch info
    batch_info_file = output_dir / "choose_prompts_batch_info.json"
    if not batch_info_file.exists():
        print("Error: choose_prompts_batch_info.json not found!")
        print("Run choose_best_prompts.py first.")
        exit(1)
    
    with open(batch_info_file) as f:
        batch_info = json.load(f)
    
    batch_id = batch_info["batch_id"]
    print(f"Checking batch {batch_id}...")
    
    # Get batch status
    message_batch = client.messages.batches.retrieve(batch_id)
    
    print(f"\nStatus: {message_batch.processing_status}")
    print("Request counts:")
    print(f"  Total: {message_batch.request_counts.processing + message_batch.request_counts.succeeded + message_batch.request_counts.errored + message_batch.request_counts.canceled + message_batch.request_counts.expired}")
    print(f"  Succeeded: {message_batch.request_counts.succeeded}")
    print(f"  Errored: {message_batch.request_counts.errored}")
    print(f"  Processing: {message_batch.request_counts.processing}")
    print(f"  Canceled: {message_batch.request_counts.canceled}")
    print(f"  Expired: {message_batch.request_counts.expired}")
    
    if message_batch.processing_status != "ended":
        print("\nBatch is still processing. Check back later.")
        exit(0)
    
    print("\nBatch complete! Retrieving results...")
    
    # Load the original entries
    entries_file = output_dir / "entries_needing_prompt_selection.json"
    with open(entries_file) as f:
        entries = json.load(f)
    
    print(f"Loaded {len(entries)} original entries")
    
    # Parse results and match to entries
    chosen_prompts = {}
    failed_results = []
    
    # Iterate through results using the SDK
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        
        # Extract index from custom_id (format: choose_prompt_00000)
        idx = int(custom_id.split("_")[-1])
        
        if result.result.type == "succeeded":
            # Extract the chosen prompt from the response
            content_block = result.result.message.content[0]
            if hasattr(content_block, 'text'):
                response_text = content_block.text.strip()
                
                # Clean up any markdown formatting
                if response_text.startswith("```"):
                    lines = response_text.split("\n")
                    response_text = "\n".join(lines[1:-1] if len(lines) > 2 else lines).strip()
                
                original_title = entries[idx]["original_title"]
                chosen_prompts[original_title] = response_text
            else:
                failed_results.append({
                    "custom_id": custom_id,
                    "index": idx,
                    "original_title": entries[idx]["original_title"],
                    "error": f"Content block has no text attribute: {type(content_block)}"
                })
        else:
            # Record failure
            error_msg = str(result.result.error) if hasattr(result.result, 'error') else "Unknown error"
            failed_results.append({
                "custom_id": custom_id,
                "index": idx,
                "original_title": entries[idx]["original_title"],
                "error": error_msg
            })
    
    print(f"\nSuccessfully chose prompts for {len(chosen_prompts)} entries")
    if failed_results:
        print(f"Failed: {len(failed_results)} entries")
    
    # Save chosen prompts
    output_file = output_dir / "chosen_prompts.json"
    with open(output_file, "w") as f:
        json.dump(chosen_prompts, f, indent=2)
    
    print(f"\nSaved chosen prompts to {output_file.name}")
    
    if failed_results:
        failed_file = output_dir / "failed_prompt_choices.json"
        with open(failed_file, "w") as f:
            json.dump(failed_results, f, indent=2)
        print(f"Saved {len(failed_results)} failures to {failed_file.name}")
    
    print("\n✓ Done! Re-run merge_topics.py to use these chosen prompts.")


if __name__ == "__main__":
    main()
