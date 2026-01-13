#!/usr/bin/env python3
"""Filter the merged dataset based on coherence scores.

Usage:
    python filter_by_coherence.py <min_score>

Example:
    python filter_by_coherence.py 9

This keeps only entries with coherence score >= min_score (on a 0-10 scale).
For the published dataset, we use min_score=9.
"""

import json
from pathlib import Path
import sys


def main():
    output_dir = Path(__file__).parent / "outputs"
    
    # Check if threshold was provided
    if len(sys.argv) < 2:
        print("Usage: python filter_by_coherence.py <min_score>")
        print("Example: python filter_by_coherence.py 9")
        print("\nThis will keep only entries with coherence score >= min_score")
        exit(1)
    
    try:
        min_score = float(sys.argv[1])
    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid number")
        exit(1)
    
    if min_score < 0 or min_score > 10:
        print("Error: min_score must be between 0 and 10")
        exit(1)
    
    # Load coherence scores
    scores_file = output_dir / "coherence_scores.json"
    if not scores_file.exists():
        print("Error: coherence_scores.json not found!")
        print("Run check_coherence_scores.py first.")
        exit(1)
    
    with open(scores_file) as f:
        coherence_scores = json.load(f)
    
    # Load merged topics
    merged_file = output_dir / "generated_topics_merged.json"
    with open(merged_file) as f:
        merged_data = json.load(f)
    
    topics = merged_data["topics"]
    
    print(f"Loaded {len(topics)} topics from {merged_file.name}")
    print(f"Loaded {len(coherence_scores)} coherence scores from {scores_file.name}")
    
    # Create a score lookup dict
    scores_by_title = {s["original_title"]: s["score"] for s in coherence_scores if s["score"] is not None}
    
    print(f"\nFiltering to keep entries with score >= {min_score}...")
    
    # Filter topics
    filtered_topics = []
    removed_topics = []
    
    for topic in topics:
        title = topic["original_title"]
        score = scores_by_title.get(title)
        
        if score is None:
            # No score available - be conservative and remove it
            removed_topics.append({
                "topic": topic,
                "reason": "no_score",
                "score": None
            })
        elif score >= min_score:
            filtered_topics.append(topic)
        else:
            removed_topics.append({
                "topic": topic,
                "reason": "low_score",
                "score": score
            })
    
    print(f"\nResults:")
    print(f"  Kept: {len(filtered_topics)} topics")
    print(f"  Removed: {len(removed_topics)} topics")
    
    if removed_topics:
        low_score_count = sum(1 for r in removed_topics if r["reason"] == "low_score")
        no_score_count = sum(1 for r in removed_topics if r["reason"] == "no_score")
        print(f"    - Low score: {low_score_count}")
        print(f"    - No score: {no_score_count}")
    
    # Calculate new statistics
    total_labels = sum(len(t["labels"]) for t in filtered_topics)
    avg_labels = total_labels / len(filtered_topics) if filtered_topics else 0
    
    # Create output
    output = {
        "topics": filtered_topics,
        "metadata": {
            **merged_data["metadata"],
            "filtered_by_coherence": True,
            "min_coherence_score": min_score,
            "topics_before_filtering": len(topics),
            "topics_after_filtering": len(filtered_topics),
            "topics_removed": len(removed_topics),
            "total_labels_after_filtering": total_labels,
            "average_labels_per_topic_after_filtering": avg_labels
        }
    }
    
    # Save filtered dataset
    output_file = output_dir / f"generated_topics_filtered_min{min_score}.json"
    print(f"\nSaving filtered dataset to {output_file.name}...")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Saved {len(filtered_topics)} topics to {output_file.name}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Also save the removed topics for inspection
    if removed_topics:
        removed_file = output_dir / f"removed_topics_min{min_score}.json"
        with open(removed_file, "w") as f:
            json.dump(removed_topics, f, indent=2)
        print(f"Saved {len(removed_topics)} removed topics to {removed_file.name}")
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
