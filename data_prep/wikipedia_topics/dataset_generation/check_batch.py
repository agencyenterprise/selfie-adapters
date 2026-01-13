#!/usr/bin/env python3
"""Check batch status and retrieve results from generation.

Usage:
    python check_batch.py

This script checks the status of a batch submitted via generate_all.py.
When the batch completes, it retrieves and parses the results.
"""

import json
import time
from pathlib import Path
from anthropic import Anthropic


def main():
    # Initialize Anthropic client
    client = Anthropic()
    
    # Load batch info
    output_dir = Path(__file__).parent / "outputs"
    batch_info_file = output_dir / "batch_info.json"
    
    if not batch_info_file.exists():
        print("Error: No batch_info.json found. Run generate_all.py first.")
        exit(1)
    
    with open(batch_info_file) as f:
        batch_info = json.load(f)
    
    batch_id = batch_info["batch_id"]
    print(f"Checking batch ID: {batch_id}")
    print(f"Submitted at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(batch_info['submitted_at']))}")
    print(f"Number of requests: {batch_info['num_requests']}")
    print()
    
    # Check batch status
    message_batch = client.messages.batches.retrieve(batch_id)
    
    print(f"Status: {message_batch.processing_status}")
    print(f"Requests: {message_batch.request_counts}")
    
    if message_batch.processing_status == "ended":
        print("\n✓ Batch processing completed!")
        
        # Retrieve and parse results
        print("\nRetrieving results...")
        
        results = []
        failed = []
        
        # Iterate through all results
        for result in client.messages.batches.results(batch_id):
            if result.result.type == "succeeded":
                try:
                    response_text = result.result.message.content[0].text
                    
                    # Strip markdown code fences if present
                    clean_text = response_text.strip()
                    if clean_text.startswith("```json"):
                        clean_text = clean_text[7:]
                    if clean_text.startswith("```"):
                        clean_text = clean_text[3:]
                    if clean_text.endswith("```"):
                        clean_text = clean_text[:-3]
                    clean_text = clean_text.strip()
                    
                    # Parse JSON
                    parsed = json.loads(clean_text)
                    
                    # Handle both single object and array responses
                    if isinstance(parsed, list):
                        results.extend(parsed)
                    else:
                        results.append(parsed)
                        
                except Exception as e:
                    print(f"Error parsing result for {result.custom_id}: {e}")
                    failed.append({
                        "custom_id": result.custom_id,
                        "error": str(e),
                        "raw_response": response_text[:200]
                    })
            else:
                print(f"Failed request: {result.custom_id}")
                failed.append({
                    "custom_id": result.custom_id,
                    "error": result.result.error.message if hasattr(result.result, 'error') else "Unknown error"
                })
        
        print(f"\n✓ Successfully parsed {len(results)} results")
        if failed:
            print(f"✗ Failed to parse {len(failed)} results")
        
        # Save results
        output_file = output_dir / "generated_topics.json"
        with open(output_file, "w") as f:
            json.dump({
                "topics": results,
                "metadata": {
                    "total": len(results),
                    "batch_id": batch_id,
                    "generated_at": time.time()
                }
            }, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")
        
        # Save failed results if any
        if failed:
            failed_file = output_dir / "failed_results.json"
            with open(failed_file, "w") as f:
                json.dump(failed, f, indent=2)
            print(f"✗ Failed results saved to {failed_file}")
        
        # Print some sample results
        print("\n" + "=" * 80)
        print("Sample results:")
        print("=" * 80)
        for i, topic in enumerate(results[:3], 1):
            print(f"\n{i}. {topic['original_title']}")
            print(f"   Prompt: {topic['prompt']}")
            print(f"   Labels ({len(topic['labels'])}):")
            for label in topic['labels']:
                print(f"     - {label}")
    
    elif message_batch.processing_status == "in_progress":
        elapsed = time.time() - batch_info['submitted_at']
        print(f"\n⏳ Batch is still processing (elapsed: {elapsed/60:.1f} minutes)")
        print("Check again in a few minutes...")
    
    elif message_batch.processing_status == "canceling":
        print("\n⚠ Batch is being canceled...")
    
    elif message_batch.processing_status == "canceled":
        print("\n✗ Batch was canceled")
    
    else:
        print(f"\n? Unknown status: {message_batch.processing_status}")


if __name__ == "__main__":
    main()
