#!/usr/bin/env python3
"""
Filter TwoHopFact dataset by querying Llama 3.1 8B Instruct and checking correctness.

This script:
1. Loads the TwoHopFact dataset
2. Formats each question with 1-shot prompting (no CoT allowed)
3. Queries vLLM server with large async batches
4. Checks if model's answer matches any of the e3.aliases (final answer)
5. ALSO checks if model can correctly answer e2 (bridge entity) when asked directly
6. Saves the filtered dataset (only examples where BOTH e2 and e3 are correct)

The e2 filter prevents "shortcut" answers where the model guesses e3 correctly
based on surface patterns (e.g., guessing "Morocco" from "Moroccan" in the title)
without actually knowing the bridge entity (e.g., the author "Hicham Nostik").
"""
import asyncio
import time
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
import httpx
from datasets import load_dataset, Dataset
from tqdm.asyncio import tqdm as atqdm


class CustomLlamaInference:
    """Custom async wrapper with 1-shot prompting for TwoHopFact filtering."""
    
    def __init__(self, base_url: str = "http://localhost:8000/v1", timeout: float = 300.0):
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        
        # Fixed 1-shot example (Rainbow Six Siege -> Ottawa)
        self.few_shot_example = {
            "category": "city",
            "prompt": "The capital of the country of origin of Tom Clancy's Rainbow Six Siege is",
            "answer": "Ottawa"
        }
        
        # Generation params - temperature 0 for deterministic answers
        self.gen_params = {
            "temperature": 0.0,
            "max_tokens": 50,  # Short answers only
            "top_p": 1.0,
        }
    
    async def close(self):
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    def format_messages(self, category: str, prompt: str) -> List[Dict[str, str]]:
        """
        Format messages with 1-shot prompting for e3 (final answer).
        
        Args:
            category: The category (e.g., "city", "person", "human")
            prompt: The actual prompt to complete
            
        Returns:
            List of message dicts for chat API
        """
        # Note: We map "person" to "human" to match the user's example
        display_category = "human" if category == "person" else category
        
        messages = [
            {
                "role": "user",
                "content": f"Complete the following statement with only the name of a {self.few_shot_example['category']}. "
                          f"If you don't know, make your best guess. {self.few_shot_example['prompt']}"
            },
            {
                "role": "assistant",
                "content": self.few_shot_example['answer']
            },
            {
                "role": "user",
                "content": f"Complete the following statement with only the name of a {display_category}. "
                          f"If you don't know, make your best guess. {prompt}"
            }
        ]
        
        return messages
    
    def format_messages_for_e2(self, category: str, prompt: str) -> List[Dict[str, str]]:
        """
        Format messages with 1-shot prompting for e2 (bridge entity).
        
        Uses a different 1-shot example appropriate for asking about the
        intermediate entity (e.g., "The author of X is" -> person name).
        
        Args:
            category: The category of e2 (e.g., "person", "city", "company")
            prompt: The r1(e1).prompt to complete (e.g., "The author of the novel X is")
            
        Returns:
            List of message dicts for chat API
        """
        # Map category to human-readable form
        display_category = "human" if category == "person" else category
        
        # 1-shot example for e2: asking about an author
        # "The author of the novel 1984 is" -> "George Orwell"
        e2_example = {
            "category": "human",
            "prompt": "The author of the novel 1984 is",
            "answer": "George Orwell"
        }
        
        messages = [
            {
                "role": "user",
                "content": f"Complete the following statement with only the name of a {e2_example['category']}. "
                          f"If you don't know, make your best guess. {e2_example['prompt']}"
            },
            {
                "role": "assistant",
                "content": e2_example['answer']
            },
            {
                "role": "user",
                "content": f"Complete the following statement with only the name of a {display_category}. "
                          f"If you don't know, make your best guess. {prompt}"
            }
        ]
        
        return messages
    
    async def generate_single(self, category: str, prompt: str, is_e2: bool = False) -> str:
        """
        Generate answer for a single prompt.
        
        Args:
            category: The entity category
            prompt: The prompt to complete
            is_e2: If True, use e2-specific formatting (for bridge entity)
        """
        if is_e2:
            messages = self.format_messages_for_e2(category, prompt)
        else:
            messages = self.format_messages(category, prompt)
        
        payload = {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": messages,
            **self.gen_params
        }
        
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        answer = result["choices"][0]["message"]["content"].strip()
        
        return answer
    
    async def generate_batch(self, examples: List[Dict[str, str]], 
                            batch_size: int = 100) -> List[str]:
        """
        Generate answers for multiple examples with progress bar.
        
        Args:
            examples: List of dicts with 'category' and 'prompt' keys
            batch_size: Number of concurrent requests (adjust based on memory)
            
        Returns:
            List of generated answers
        """
        all_answers = []
        
        # Process in chunks to avoid overwhelming the server
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            
            tasks = [
                self.generate_single(ex['category'], ex['prompt'])
                for ex in batch
            ]
            
            answers = await asyncio.gather(*tasks)
            all_answers.extend(answers)
        
        return all_answers


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    # Remove common punctuation and convert to lowercase
    answer = answer.lower().strip()
    # Remove trailing punctuation
    answer = answer.rstrip('.,!?;:')
    return answer


def check_answer_match(model_answer: str, aliases_str: str) -> bool:
    """
    Check if model's answer matches any of the ground truth aliases.
    
    Args:
        model_answer: The answer from the model
        aliases_str: String representation of aliases (e.g., "(('Paris', 'Paree'),)")
        
    Returns:
        True if match found, False otherwise
    """
    import ast
    
    normalized_model = normalize_answer(model_answer)
    
    # Parse the string representation of aliases
    try:
        aliases = ast.literal_eval(aliases_str)
        # Flatten nested tuples - structure is (('alias1', 'alias2'),)
        if aliases and len(aliases) > 0:
            flat_aliases = aliases[0] if isinstance(aliases[0], tuple) else aliases
        else:
            return False
    except (ValueError, SyntaxError):
        # If parsing fails, return False
        return False
    
    # Check for exact match or substring match
    for alias in flat_aliases:
        normalized_alias = normalize_answer(alias)
        
        # Exact match
        if normalized_model == normalized_alias:
            return True
        
        # Model answer contains the alias (handles cases like "The answer is Paris" -> "Paris")
        if normalized_alias in normalized_model:
            return True
        
        # Alias contains model answer (handles cases where model gives short form)
        if normalized_model in normalized_alias:
            return True
    
    return False


async def filter_dataset(
    dataset: Dataset,
    inferencer: CustomLlamaInference,
    batch_size: int = 100,
    max_examples: int = None
) -> Tuple[Dataset, Dict[str, Any]]:
    """
    Filter dataset by checking model predictions for BOTH e2 and e3.
    
    An example passes the filter only if the model can:
    1. Correctly answer e3 (final answer) when given the two-hop prompt
    2. Correctly answer e2 (bridge entity) when asked directly via r1(e1).prompt
    
    This prevents "shortcut" answers where the model guesses e3 correctly
    based on surface patterns without knowing the bridge entity e2.
    
    Args:
        dataset: The TwoHopFact dataset
        inferencer: The inference client
        batch_size: Batch size for async requests
        max_examples: Optional limit for testing (None = process all)
        
    Returns:
        Tuple of (filtered_dataset, statistics_dict)
    """
    print(f"\n{'='*80}")
    print("Starting dataset filtering (checking BOTH e2 and e3)")
    print(f"{'='*80}\n")
    
    # Limit examples if specified
    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
        print(f"Processing first {len(dataset)} examples (limited for testing)\n")
    else:
        print(f"Processing all {len(dataset)} examples\n")
    
    # =========================================================================
    # PHASE 1: Check e3 answers (final answer from two-hop prompt)
    # =========================================================================
    print("="*60)
    print("PHASE 1: Checking e3 (final answer) predictions")
    print("="*60 + "\n")
    
    # Prepare e3 examples for inference
    print("Preparing e3 examples...")
    e3_examples = []
    for item in dataset:
        e3_examples.append({
            'category': item['e3.rough_category'],
            'prompt': item['r2(r1(e1)).prompt']
        })
    
    # Generate e3 answers
    print(f"\nQuerying model for e3 (batch_size={batch_size})...")
    start_time = time.time()
    
    all_e3_answers = []
    total_batches = (len(e3_examples) + batch_size - 1) // batch_size
    
    for i in range(0, len(e3_examples), batch_size):
        batch = e3_examples[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        print(f"E3 batch {batch_num}/{total_batches} "
              f"(examples {i+1}-{min(i+batch_size, len(e3_examples))})...")
        
        tasks = [
            inferencer.generate_single(ex['category'], ex['prompt'], is_e2=False)
            for ex in batch
        ]
        
        answers = await asyncio.gather(*tasks)
        all_e3_answers.extend(answers)
    
    e3_elapsed = time.time() - start_time
    print(f"\nE3 completed in {e3_elapsed:.2f}s ({e3_elapsed/len(e3_examples):.3f}s per example)")
    
    # Check e3 correctness
    print(f"Checking e3 answers against ground truth...")
    e3_correct = []
    for idx, (item, model_answer) in enumerate(zip(dataset, all_e3_answers)):
        is_correct = check_answer_match(model_answer, item['e3.aliases'])
        e3_correct.append(is_correct)
    
    e3_correct_count = sum(e3_correct)
    print(f"E3 accuracy: {e3_correct_count}/{len(dataset)} ({100*e3_correct_count/len(dataset):.2f}%)")
    
    # =========================================================================
    # PHASE 2: Check e2 answers (bridge entity from r1(e1).prompt)
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 2: Checking e2 (bridge entity) predictions")
    print("="*60 + "\n")
    
    # Prepare e2 examples for inference
    print("Preparing e2 examples...")
    e2_examples = []
    for item in dataset:
        e2_examples.append({
            'category': item['e2.rough_category'],
            'prompt': item['r1(e1).prompt']
        })
    
    # Generate e2 answers
    print(f"\nQuerying model for e2 (batch_size={batch_size})...")
    start_time = time.time()
    
    all_e2_answers = []
    total_batches = (len(e2_examples) + batch_size - 1) // batch_size
    
    for i in range(0, len(e2_examples), batch_size):
        batch = e2_examples[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        print(f"E2 batch {batch_num}/{total_batches} "
              f"(examples {i+1}-{min(i+batch_size, len(e2_examples))})...")
        
        tasks = [
            inferencer.generate_single(ex['category'], ex['prompt'], is_e2=True)
            for ex in batch
        ]
        
        answers = await asyncio.gather(*tasks)
        all_e2_answers.extend(answers)
    
    e2_elapsed = time.time() - start_time
    print(f"\nE2 completed in {e2_elapsed:.2f}s ({e2_elapsed/len(e2_examples):.3f}s per example)")
    
    # Check e2 correctness
    print(f"Checking e2 answers against ground truth...")
    e2_correct = []
    for idx, (item, model_answer) in enumerate(zip(dataset, all_e2_answers)):
        is_correct = check_answer_match(model_answer, item['e2.aliases'])
        e2_correct.append(is_correct)
    
    e2_correct_count = sum(e2_correct)
    print(f"E2 accuracy: {e2_correct_count}/{len(dataset)} ({100*e2_correct_count/len(dataset):.2f}%)")
    
    # =========================================================================
    # PHASE 3: Combine filters - require BOTH e2 and e3 correct
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 3: Combining filters (require BOTH e2 AND e3 correct)")
    print("="*60 + "\n")
    
    # Find examples where BOTH e2 and e3 are correct
    correct_indices = []
    stats = {
        'total': len(dataset),
        'e3_correct': e3_correct_count,
        'e2_correct': e2_correct_count,
        'both_correct': 0,
        'e3_only': 0,  # Got e3 right but e2 wrong (shortcut cases!)
        'e2_only': 0,  # Got e2 right but e3 wrong
        'both_wrong': 0,
        'accuracy_e3': 100 * e3_correct_count / len(dataset),
        'accuracy_e2': 100 * e2_correct_count / len(dataset),
        'accuracy_both': 0.0,
        'examples': []  # Store examples for inspection
    }
    
    for idx, item in enumerate(dataset):
        e3_ok = e3_correct[idx]
        e2_ok = e2_correct[idx]
        
        if e3_ok and e2_ok:
            correct_indices.append(idx)
            stats['both_correct'] += 1
        elif e3_ok and not e2_ok:
            stats['e3_only'] += 1  # These are the shortcut cases!
        elif not e3_ok and e2_ok:
            stats['e2_only'] += 1
        else:
            stats['both_wrong'] += 1
        
        # Store examples for inspection (various categories)
        import ast
        try:
            e3_aliases = ast.literal_eval(item['e3.aliases'])
            e3_alias_list = e3_aliases[0] if e3_aliases and len(e3_aliases) > 0 else ()
        except (ValueError, SyntaxError):
            e3_alias_list = ()
        
        try:
            e2_aliases = ast.literal_eval(item['e2.aliases'])
            e2_alias_list = e2_aliases[0] if e2_aliases and len(e2_aliases) > 0 else ()
        except (ValueError, SyntaxError):
            e2_alias_list = ()
        
        # Store up to 5 examples of each type
        example_type = None
        if e3_ok and e2_ok:
            example_type = 'both_correct'
        elif e3_ok and not e2_ok:
            example_type = 'shortcut'  # These are interesting!
        elif not e3_ok and e2_ok:
            example_type = 'e2_only'
        else:
            example_type = 'both_wrong'
        
        type_count = len([e for e in stats['examples'] if e.get('example_type') == example_type])
        if type_count < 5:
            stats['examples'].append({
                'uid': item['uid'],
                'category': item['category'],
                'e3_prompt': item['r2(r1(e1)).prompt'],
                'e2_prompt': item['r1(e1).prompt'],
                'e3_model_answer': all_e3_answers[idx],
                'e2_model_answer': all_e2_answers[idx],
                'e3_correct_answer': item['e3.value'],
                'e2_correct_answer': item['e2.value'],
                'e3_aliases': e3_alias_list,
                'e2_aliases': e2_alias_list[:5] if len(e2_alias_list) > 5 else e2_alias_list,  # Truncate long alias lists
                'e3_correct': e3_ok,
                'e2_correct': e2_ok,
                'example_type': example_type
            })
    
    stats['accuracy_both'] = 100 * stats['both_correct'] / stats['total']
    
    # Create filtered dataset
    print(f"Creating filtered dataset...")
    filtered_dataset = dataset.select(correct_indices)
    
    return filtered_dataset, stats, all_e2_answers, all_e3_answers


def print_statistics(stats: Dict[str, Any]):
    """Print filtering statistics with e2/e3 breakdown."""
    print(f"\n{'='*80}")
    print("FILTERING RESULTS")
    print(f"{'='*80}\n")
    
    print(f"Total examples: {stats['total']}")
    print(f"\nE3 (final answer) correct: {stats['e3_correct']} ({stats['accuracy_e3']:.2f}%)")
    print(f"E2 (bridge entity) correct: {stats['e2_correct']} ({stats['accuracy_e2']:.2f}%)")
    print(f"\nBreakdown:")
    print(f"  ✓ Both e2 AND e3 correct: {stats['both_correct']} ({stats['accuracy_both']:.2f}%)")
    print(f"  ⚠ E3 only (SHORTCUT!):    {stats['e3_only']} ({100*stats['e3_only']/stats['total']:.2f}%)")
    print(f"  - E2 only:                {stats['e2_only']} ({100*stats['e2_only']/stats['total']:.2f}%)")
    print(f"  ✗ Both wrong:             {stats['both_wrong']} ({100*stats['both_wrong']/stats['total']:.2f}%)")
    
    print(f"\n{'='*80}")
    print("SHORTCUT CASES (got e3 right but e2 wrong - these are filtered OUT)")
    print(f"{'='*80}\n")
    
    shortcut_examples = [e for e in stats['examples'] if e.get('example_type') == 'shortcut']
    if shortcut_examples:
        for ex in shortcut_examples[:5]:
            print(f"Category: {ex['category']}")
            print(f"E2 prompt: {ex['e2_prompt']}")
            print(f"  Model said: {ex['e2_model_answer']}")
            print(f"  Should be:  {ex['e2_correct_answer']} (aliases: {ex['e2_aliases']})")
            print(f"E3 prompt: {ex['e3_prompt']}")
            print(f"  Model said: {ex['e3_model_answer']}")
            print(f"  Should be:  {ex['e3_correct_answer']}")
            print(f"⚠ SHORTCUT: Got final answer right without knowing bridge entity!\n")
    else:
        print("No shortcut cases found in sample.\n")
    
    print(f"{'-'*80}")
    print("Sample PASSING examples (both e2 AND e3 correct):")
    print(f"{'-'*80}\n")
    for ex in [e for e in stats['examples'] if e.get('example_type') == 'both_correct'][:3]:
        print(f"Category: {ex['category']}")
        print(f"E2: \"{ex['e2_prompt']}\" → \"{ex['e2_model_answer']}\" ✓")
        print(f"E3: \"{ex['e3_prompt']}\" → \"{ex['e3_model_answer']}\" ✓")
        print(f"Ground truth: e2={ex['e2_correct_answer']}, e3={ex['e3_correct_answer']}")
        print()


def save_filtered_dataset_as_json(filtered_dataset: Dataset, 
                                  original_dataset: Dataset,
                                  stats: Dict[str, Any],
                                  all_e2_answers: List[str],
                                  all_e3_answers: List[str],
                                  output_path: Path):
    """
    Save filtered dataset in JSON format matching the example_code format.
    
    Args:
        filtered_dataset: The filtered dataset (both e2 and e3 correct)
        original_dataset: The original full dataset
        stats: Statistics dict
        all_e2_answers: All e2 model answers (indexed by original dataset)
        all_e3_answers: All e3 model answers (indexed by original dataset)
        output_path: Path to save the JSON file
    """
    import ast
    
    # Build a mapping of uid -> (e2_answer, e3_answer)
    uid_to_e2 = {}
    uid_to_e3 = {}
    for idx, item in enumerate(original_dataset):
        uid_to_e2[item['uid']] = all_e2_answers[idx]
        uid_to_e3[item['uid']] = all_e3_answers[idx]
    
    json_data = {
        "summary": {
            "total_questions": stats['total'],
            "e3_correct": stats['e3_correct'],
            "e2_correct": stats['e2_correct'],
            "both_correct": stats['both_correct'],
            "shortcut_cases": stats['e3_only'],
            "accuracy_e3": stats['accuracy_e3'],
            "accuracy_e2": stats['accuracy_e2'],
            "accuracy_both": stats['accuracy_both'],
        },
        "filtered_results": []
    }
    
    for item in filtered_dataset:
        uid = item['uid']
        
        # Parse e3 aliases (answer aliases)
        try:
            e3_aliases = ast.literal_eval(item['e3.aliases'])
            e3_alias_list = list(e3_aliases[0]) if e3_aliases and len(e3_aliases) > 0 else []
        except (ValueError, SyntaxError):
            e3_alias_list = []
        
        # Parse e2 aliases (bridge entity aliases) - IMPORTANT for interpretability!
        try:
            e2_aliases = ast.literal_eval(item['e2.aliases'])
            e2_alias_list = list(e2_aliases[0]) if e2_aliases and len(e2_aliases) > 0 else []
        except (ValueError, SyntaxError):
            e2_alias_list = []
        
        result = {
            "id": str(uid),
            "question": item['r2(r1(e1)).prompt'],
            "entity_type": item['e3.rough_category'],
            "correct_answer": item['e3.value'],
            "aliases": e3_alias_list,
            "category": item['category'],
            "fact_comp_type": item['fact_comp_type'],
            # Include entity info for interpretability work
            "e1_value": item['e1.value'],
            "e2_value": item['e2.value'],  # This is the "bridge entity"!
            "e2_aliases": e2_alias_list,  # Alternative names for the bridge entity
            "e3_value": item['e3.value'],
            # Include the prompts used
            "e2_prompt": item['r1(e1).prompt'],
            "e3_prompt": item['r2(r1(e1)).prompt'],
            # Include model's answers (useful for verification)
            "model_e2_answer": uid_to_e2.get(uid, ""),
            "model_e3_answer": uid_to_e3.get(uid, ""),
        }
        
        json_data['filtered_results'].append(result)
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Saved {len(json_data['filtered_results'])} filtered examples to JSON")


async def main():
    """Main filtering pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter TwoHopFact dataset")
    parser.add_argument("--output-dir", type=str, default="./filtered_data",
                        help="Output directory for filtered dataset")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Batch size for async requests (adjust based on GPU memory)")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Maximum examples to process (None = all)")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1",
                        help="URL of the vLLM server")
    args = parser.parse_args()
    
    # Configuration
    BATCH_SIZE = args.batch_size
    MAX_EXAMPLES = args.max_examples
    OUTPUT_DIR = Path(args.output_dir)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading TwoHopFact dataset...")
    dataset = load_dataset("soheeyang/TwoHopFact")
    train_data = dataset['train']
    
    # Filter dataset (checks BOTH e2 and e3)
    async with CustomLlamaInference(base_url=args.vllm_url) as inferencer:
        filtered_dataset, stats, all_e2_answers, all_e3_answers = await filter_dataset(
            train_data,
            inferencer,
            batch_size=BATCH_SIZE,
            max_examples=MAX_EXAMPLES
        )
    
    # Print results
    print_statistics(stats)
    
    # Save filtered dataset (Arrow format)
    output_path = OUTPUT_DIR / "filtered_dataset"
    print(f"\n{'='*80}")
    print(f"Saving filtered dataset to {output_path}")
    print(f"{'='*80}\n")
    
    filtered_dataset.save_to_disk(str(output_path))
    
    # Save statistics
    stats_path = OUTPUT_DIR / "filtering_stats.json"
    with open(stats_path, 'w') as f:
        # Remove examples for cleaner stats file
        stats_to_save = {k: v for k, v in stats.items() if k != 'examples'}
        json.dump(stats_to_save, f, indent=2)
    
    print(f"Statistics saved to {stats_path}")
    
    # Also save in JSON format matching the example_code format
    json_path = OUTPUT_DIR / "filtered_dataset.json"
    print(f"\nSaving JSON format to {json_path}...")
    save_filtered_dataset_as_json(
        filtered_dataset, train_data, stats, 
        all_e2_answers, all_e3_answers, json_path
    )
    
    print(f"\n{'='*80}")
    print(f"Filtering complete!")
    print(f"Total examples: {stats['total']}")
    print(f"E3-only correct: {stats['e3_correct']} (these include shortcuts)")
    print(f"E2-only correct: {stats['e2_correct']}")
    print(f"BOTH correct: {stats['both_correct']} (these are in filtered dataset)")
    print(f"Shortcut cases filtered out: {stats['e3_only']}")
    print(f"\nFiltered dataset: {len(filtered_dataset)} examples")
    print(f"Arrow format: {output_path}")
    print(f"JSON format: {json_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    print("\n" + "!"*80)
    print("IMPORTANT: Make sure the vLLM server is running!")
    print("Start it with: python start_vllm_server.py")
    print("Check status with: python check_server.py")
    print("!"*80 + "\n")
    
    asyncio.run(main())

