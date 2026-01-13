#!/usr/bin/env python3
"""Convert filtered JSON to JSONL with train/val splits.

Usage:
    python create_jsonl_splits.py

This creates the final dataset with 90% train / 10% val split.
Output is ready for upload to HuggingFace.
"""

import json
import random
from pathlib import Path


def main():
    output_dir = Path(__file__).parent / "outputs"
    
    # Load the filtered dataset
    input_file = output_dir / "generated_topics_filtered_min9.0.json"
    
    if not input_file.exists():
        print(f"Error: {input_file.name} not found!")
        print("Run filter_by_coherence.py 9 first.")
        exit(1)
    
    print(f"Loading {input_file.name}...")
    
    with open(input_file) as f:
        data = json.load(f)
    
    topics = data["topics"]
    print(f"Loaded {len(topics)} topics")
    
    # Shuffle topics for random split
    print("\nShuffling topics for random train/val split...")
    random.seed(42)  # For reproducibility
    shuffled_topics = topics.copy()
    random.shuffle(shuffled_topics)
    
    # Calculate split sizes (90% train, 10% val)
    total = len(shuffled_topics)
    train_size = int(total * 0.9)
    val_size = total - train_size
    
    print(f"\nSplit sizes:")
    print(f"  Train: {train_size} ({train_size/total*100:.1f}%)")
    print(f"  Val:   {val_size} ({val_size/total*100:.1f}%)")
    
    # Assign splits
    train_topics = shuffled_topics[:train_size]
    val_topics = shuffled_topics[train_size:]
    
    # Add split field to each entry
    for topic in train_topics:
        topic["split"] = "train"
    
    for topic in val_topics:
        topic["split"] = "val"
    
    # Combine back together (train first, then val)
    all_topics = train_topics + val_topics
    
    # Write as JSONL
    output_file = output_dir / "wikipedia_vital_articles_level5_dataset.jsonl"
    print(f"\nWriting to {output_file.name}...")
    
    with open(output_file, "w") as f:
        for topic in all_topics:
            f.write(json.dumps(topic) + "\n")
    
    print(f"✓ Wrote {len(all_topics)} entries to {output_file.name}")
    
    # Verify split counts
    train_count = sum(1 for t in all_topics if t["split"] == "train")
    val_count = sum(1 for t in all_topics if t["split"] == "val")
    print(f"\nVerification:")
    print(f"  Train entries: {train_count}")
    print(f"  Val entries:   {val_count}")
    print(f"  Total:         {train_count + val_count}")
    
    # Calculate file size
    file_size_mb = output_file.stat().st_size / 1024 / 1024
    print(f"\nFile size: {file_size_mb:.1f} MB")
    
    # Show a few sample entries
    print("\n" + "="*80)
    print("Sample entries:")
    print("="*80)
    
    for i, topic in enumerate(all_topics[:2], 1):
        print(f"\n{i}. {topic['original_title']} (split: {topic['split']})")
        print(f"   Prompt: {topic['prompt']}")
        print(f"   Labels ({len(topic['labels'])}): {', '.join(topic['labels'][:3])}...")
    
    print("\n✓ Done! Ready for HuggingFace upload.")


if __name__ == "__main__":
    main()
