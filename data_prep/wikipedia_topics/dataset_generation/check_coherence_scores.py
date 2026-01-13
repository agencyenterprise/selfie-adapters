#!/usr/bin/env python3
"""Check batch status and retrieve coherence scores.

Usage:
    python check_coherence_scores.py

After results are retrieved, use filter_by_coherence.py to filter the dataset.
"""

import json
from pathlib import Path
from anthropic import Anthropic


def main():
    # Initialize Anthropic client
    client = Anthropic()
    
    output_dir = Path(__file__).parent / "outputs"
    
    # Load batch info
    batch_info_file = output_dir / "coherence_batch_info.json"
    if not batch_info_file.exists():
        print("Error: coherence_batch_info.json not found!")
        print("Run score_coherence.py first.")
        exit(1)
    
    with open(batch_info_file) as f:
        batch_info = json.load(f)
    
    batch_id = batch_info["batch_id"]
    print(f"Checking batch {batch_id}...")
    
    # Get batch status
    message_batch = client.messages.batches.retrieve(batch_id)
    
    print(f"\nStatus: {message_batch.processing_status}")
    print(f"Request counts:")
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
    
    # Load the original merged topics
    merged_file = output_dir / "generated_topics_merged.json"
    with open(merged_file) as f:
        merged_data = json.load(f)
    
    topics = merged_data["topics"]
    print(f"Loaded {len(topics)} original topics")
    
    # Parse results and match to topics
    coherence_scores = []
    failed_results = []
    parse_errors = []
    
    # Iterate through results using the SDK
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        
        # Extract index from custom_id (format: coherence_00000)
        idx = int(custom_id.split("_")[-1])
        
        if result.result.type == "succeeded":
            # Extract the response
            content_block = result.result.message.content[0]
            if not hasattr(content_block, 'text'):
                failed_results.append({
                    "custom_id": custom_id,
                    "index": idx,
                    "original_title": topics[idx]["original_title"],
                    "error": f"Content block has no text attribute: {type(content_block)}"
                })
                continue
            
            response_text = content_block.text.strip()
            
            # Clean up any markdown formatting
            if response_text.startswith("```json"):
                response_text = response_text[7:]  # Remove ```json
            if response_text.startswith("```"):
                response_text = response_text[3:]  # Remove ```
            if response_text.endswith("```"):
                response_text = response_text[:-3]  # Remove trailing ```
            response_text = response_text.strip()
            
            try:
                # Parse the JSON response
                score_data = json.loads(response_text)
                
                coherence_scores.append({
                    "index": idx,
                    "original_title": topics[idx]["original_title"],
                    "reasoning": score_data.get("reasoning", ""),
                    "score": score_data.get("score", None)
                })
            except json.JSONDecodeError as e:
                parse_errors.append({
                    "custom_id": custom_id,
                    "index": idx,
                    "original_title": topics[idx]["original_title"],
                    "error": str(e),
                    "response_text": response_text
                })
        else:
            # Record failure
            error_msg = str(result.result.error) if hasattr(result.result, 'error') else "Unknown error"
            failed_results.append({
                "custom_id": custom_id,
                "index": idx,
                "original_title": topics[idx]["original_title"],
                "error": error_msg
            })
    
    print(f"\nSuccessfully scored {len(coherence_scores)} entries")
    if parse_errors:
        print(f"Parse errors: {len(parse_errors)} entries")
    if failed_results:
        print(f"Failed: {len(failed_results)} entries")
    
    # Save coherence scores
    output_file = output_dir / "coherence_scores.json"
    with open(output_file, "w") as f:
        json.dump(coherence_scores, f, indent=2)
    
    print(f"\nSaved coherence scores to {output_file.name}")
    
    if parse_errors:
        parse_errors_file = output_dir / "coherence_parse_errors.json"
        with open(parse_errors_file, "w") as f:
            json.dump(parse_errors, f, indent=2)
        print(f"Saved {len(parse_errors)} parse errors to {parse_errors_file.name}")
    
    if failed_results:
        failed_file = output_dir / "coherence_failed_results.json"
        with open(failed_file, "w") as f:
            json.dump(failed_results, f, indent=2)
        print(f"Saved {len(failed_results)} failures to {failed_file.name}")
    
    # Calculate statistics
    if coherence_scores:
        scores = [s["score"] for s in coherence_scores if s["score"] is not None]
        if scores:
            print("\n" + "="*80)
            print("COHERENCE SCORE STATISTICS")
            print("="*80)
            print(f"Mean score: {sum(scores) / len(scores):.2f}")
            print(f"Min score: {min(scores)}")
            print(f"Max score: {max(scores)}")
            
            # Show distribution
            print("\nScore distribution:")
            for threshold in range(0, 11):
                count = sum(1 for s in scores if s == threshold)
                if count > 0:
                    bar = "█" * (count // 10) if count >= 10 else "▪" * count
                    print(f"  {threshold:2d}: {count:5d} {bar}")
            
            # Show some low-scoring examples
            low_scores = sorted(coherence_scores, key=lambda x: x["score"] if x["score"] is not None else 999)[:10]
            print("\n" + "="*80)
            print("LOWEST SCORING ENTRIES (top 10):")
            print("="*80)
            for entry in low_scores:
                if entry["score"] is not None:
                    print(f"\nScore: {entry['score']}/10 - {entry['original_title']}")
                    print(f"Reasoning: {entry['reasoning']}")
    
    print("\n✓ Done! Use filter_by_coherence.py to filter the dataset.")


if __name__ == "__main__":
    main()
