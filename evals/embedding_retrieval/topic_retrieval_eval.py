#!/usr/bin/env python3
"""
Topic Label Evaluation via Text Embedding Retrieval

Evaluates how well text labels describe topics by:
1. Embedding all topic titles using a sentence transformer
2. For a given label, finding the nearest neighbor(s) in the embedding space
3. Computing recall@K metrics (how often the correct topic is in top K results)
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import json
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class TopicDataset(Enum):
    """Which topic dataset to use."""
    
    # 10,008 topics, 5 labels each
    TEN_THOUSAND = "keenanpepper/ten-thousand-things"
    
    # 49,637 topics, 6-20 labels each (mean ~17)
    FIFTY_THOUSAND = "keenanpepper/fifty-thousand-things"


class IndexStrategy(Enum):
    """Strategy for what text to embed and index for each topic."""
    
    # Just the Wikipedia article title (e.g., "Toshiro Mifune")
    TITLE_ONLY = "title_only"
    
    # Title + first synthetic label
    TITLE_PLUS_FIRST_LABEL = "title_plus_first_label"
    
    # Formatted document with title and all labels
    TITLE_PLUS_ALL_LABELS = "title_plus_all_labels"
    
    # Average embedding of title + all labels (each embedded separately)
    MEAN_OF_ALL = "mean_of_all"


def format_topic_document(title: str, labels: list[str], strategy: IndexStrategy) -> str:
    """
    Format a topic into a text document for embedding.
    
    Args:
        title: Wikipedia article title
        labels: List of synthetic descriptions
        strategy: Which formatting strategy to use
        
    Returns:
        Formatted text string for embedding
    """
    if strategy == IndexStrategy.TITLE_ONLY:
        return title
    
    elif strategy == IndexStrategy.TITLE_PLUS_FIRST_LABEL:
        return f"{title}: {labels[0]}"
    
    elif strategy == IndexStrategy.TITLE_PLUS_ALL_LABELS:
        # Create a rich document with title and bullet-pointed labels
        label_bullets = "\n".join(f"* {label}" for label in labels)
        return f"{title}, who/which could also be described as:\n{label_bullets}"
    
    elif strategy == IndexStrategy.MEAN_OF_ALL:
        # This case is handled specially in build_index - return title as placeholder
        return title
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


@dataclass
class TopicRetrievalConfig:
    """Configuration for topic retrieval evaluation."""
    dataset: TopicDataset = TopicDataset.FIFTY_THOUSAND
    embedding_model: str = "thenlper/gte-large"
    device: str = "cuda"
    batch_size: int = 256  # For embedding computation
    normalize_embeddings: bool = True
    index_strategy: IndexStrategy = IndexStrategy.TITLE_ONLY
    
    @property
    def dataset_name(self) -> str:
        """Get the HuggingFace dataset name."""
        return self.dataset.value


class TopicRetrievalIndex:
    """
    Efficient index for topic title embeddings with GPU-accelerated retrieval.
    
    This is NOT a heavy-duty vector database - it's designed for ~10k-50k topics
    where we can keep everything in GPU memory and use fast matrix operations.
    """
    
    def __init__(self, config: Optional[TopicRetrievalConfig] = None):
        self.config = config or TopicRetrievalConfig()
        self.device = torch.device(self.config.device)
        
        # Load the embedding model
        print(f"Loading embedding model: {self.config.embedding_model}")
        self.model = SentenceTransformer(
            self.config.embedding_model,
            device=self.config.device
        )
        
        # Will be populated by build_index()
        self.titles: list[str] = []
        self.labels: list[list[str]] = []
        self.title_embeddings: Optional[torch.Tensor] = None  # [num_topics, embed_dim]
        
    def load_dataset(self) -> None:
        """Load the topic dataset from HuggingFace."""
        print(f"Loading dataset: {self.config.dataset_name}")
        ds = load_dataset(self.config.dataset_name, split="train")
        
        self.titles = list(ds["original_title"])
        self.labels = list(ds["labels"])
        
        # Compute label count stats
        label_counts = [len(labels) for labels in self.labels]
        min_labels, max_labels = min(label_counts), max(label_counts)
        if min_labels == max_labels:
            print(f"Loaded {len(self.titles)} topics, each with {min_labels} labels")
        else:
            mean_labels = sum(label_counts) / len(label_counts)
            print(f"Loaded {len(self.titles)} topics with {min_labels}-{max_labels} labels each (mean: {mean_labels:.1f})")
    
    def _embed_texts(self, texts: list[str], show_progress: bool = True) -> torch.Tensor:
        """
        Embed a list of texts efficiently in batches.
        
        Returns:
            Tensor of shape [len(texts), embed_dim] on GPU
        """
        # sentence-transformers handles batching internally, but we want progress
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=show_progress,
            convert_to_tensor=True,
            device=self.config.device,
            normalize_embeddings=self.config.normalize_embeddings,
        )
        
        return embeddings
    
    def build_index(self) -> None:
        """Build the embedding index using the configured strategy."""
        if not self.titles:
            self.load_dataset()
        
        strategy = self.config.index_strategy
        print(f"Building index with strategy: {strategy.value}")
        
        if strategy == IndexStrategy.MEAN_OF_ALL:
            # Special case: embed title + each label separately, then average
            # Handles variable numbers of labels per topic
            print("Embedding title + all labels separately and averaging...")
            
            # Flatten all texts: title + all labels for each topic
            all_texts = []
            topic_boundaries = [0]  # Start indices for each topic's texts
            
            for title, labels in zip(self.titles, self.labels):
                all_texts.append(title)
                all_texts.extend(labels)
                topic_boundaries.append(len(all_texts))
            
            print(f"  Embedding {len(all_texts)} total texts...")
            all_embs = self._embed_texts(all_texts)
            
            # Average embeddings for each topic
            print("  Averaging embeddings per topic...")
            topic_embeddings = []
            for i in range(len(self.titles)):
                start, end = topic_boundaries[i], topic_boundaries[i + 1]
                topic_emb = all_embs[start:end].mean(dim=0)
                topic_embeddings.append(topic_emb)
            
            self.title_embeddings = torch.stack(topic_embeddings, dim=0)
            
            # Re-normalize after averaging
            if self.config.normalize_embeddings:
                self.title_embeddings = F.normalize(self.title_embeddings, p=2, dim=1)
        
        else:
            # Format documents according to strategy
            documents = [
                format_topic_document(title, labels, strategy)
                for title, labels in zip(self.titles, self.labels)
            ]
            
            # Show example document
            print(f"\nExample indexed document:\n---\n{documents[0]}\n---\n")
            
            self.title_embeddings = self._embed_texts(documents)
        
        print(f"Built index with shape: {self.title_embeddings.shape}")
        print(f"Embeddings on device: {self.title_embeddings.device}")
    
    def save_index(self, path: str | Path) -> None:
        """Save the index to disk for faster loading later."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        torch.save(self.title_embeddings.cpu(), path / "embeddings.pt")
        
        # Save metadata
        metadata = {
            "titles": self.titles,
            "labels": self.labels,
            "config": {
                "dataset": self.config.dataset.value,
                "embedding_model": self.config.embedding_model,
                "normalize_embeddings": self.config.normalize_embeddings,
                "index_strategy": self.config.index_strategy.value,
            }
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        print(f"Saved index to {path}")
    
    def load_index(self, path: str | Path) -> None:
        """Load a pre-built index from disk."""
        path = Path(path)
        
        # Load embeddings
        self.title_embeddings = torch.load(path / "embeddings.pt", weights_only=True).to(self.device)
        
        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        self.titles = metadata["titles"]
        self.labels = metadata["labels"]
        
        print(f"Loaded index from {path}")
        print(f"Index shape: {self.title_embeddings.shape}")
    
    def get_default_cache_path(self) -> Path:
        """Get the default cache path based on config."""
        # Create a unique cache name based on dataset and strategy
        dataset_name = self.config.dataset.name.lower()  # e.g., "ten_thousand"
        strategy_name = self.config.index_strategy.value  # e.g., "title_plus_all_labels"
        return Path(__file__).parent / f"index_cache_{dataset_name}_{strategy_name}"
    
    def build_or_load_index(self, cache_path: str | Path | None = None, force_rebuild: bool = False) -> None:
        """
        Build the index, or load from cache if available.
        
        Args:
            cache_path: Path to cache directory. If None, uses default based on config.
            force_rebuild: If True, rebuild even if cache exists.
        """
        if cache_path is None:
            cache_path = self.get_default_cache_path()
        else:
            cache_path = Path(cache_path)
        
        if not force_rebuild and (cache_path / "embeddings.pt").exists():
            print(f"Loading cached index from {cache_path}")
            self.load_index(cache_path)
        else:
            print(f"Building index (will cache to {cache_path})")
            self.build_index()
            self.save_index(cache_path)
    
    def query(self, text: str, k: int = 10) -> tuple[list[int], list[float]]:
        """
        Find the top-K most similar topics for a query text.
        
        Args:
            text: Query text (e.g., a generated label)
            k: Number of results to return
            
        Returns:
            (indices, scores): Lists of top-K topic indices and their similarity scores
        """
        return self.query_batch([text], k=k)
    
    def query_batch(
        self, 
        texts: list[str], 
        k: int = 10,
        show_progress: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Find the top-K most similar topics for a batch of query texts.
        
        Args:
            texts: List of query texts
            k: Number of results per query
            show_progress: Whether to show progress bar for embedding
            
        Returns:
            (indices, scores): Tensors of shape [batch_size, k]
        """
        if self.title_embeddings is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        # Embed the query texts
        query_embeddings = self._embed_texts(texts, show_progress=show_progress)
        
        # Compute similarities via matrix multiplication (embeddings are normalized)
        # query_embeddings: [batch, embed_dim]
        # title_embeddings: [num_topics, embed_dim]
        # similarities: [batch, num_topics]
        similarities = torch.mm(query_embeddings, self.title_embeddings.T)
        
        # Get top-K
        scores, indices = torch.topk(similarities, k=min(k, len(self.titles)), dim=1)
        
        return indices, scores
    
    def compute_recall_at_k(
        self,
        query_texts: list[str],
        ground_truth_indices: list[int],
        k_values: list[int] = [1, 5, 10, 20],
        batch_size: int = 512,
    ) -> dict[int, float]:
        """
        Compute recall@K for a set of queries.
        
        Args:
            query_texts: List of query texts (e.g., generated labels)
            ground_truth_indices: Index of the correct topic for each query
            k_values: List of K values to compute recall for
            batch_size: Batch size for processing queries
            
        Returns:
            Dictionary mapping K -> recall@K
        """
        max_k = max(k_values)
        n_queries = len(query_texts)
        gt_indices = torch.tensor(ground_truth_indices, device=self.device)
        
        # Process in batches
        all_top_indices = []
        
        for i in tqdm(range(0, n_queries, batch_size), desc="Computing recalls"):
            batch_texts = query_texts[i:i + batch_size]
            top_indices, _ = self.query_batch(batch_texts, k=max_k)
            all_top_indices.append(top_indices)
        
        # Concatenate all results: [n_queries, max_k]
        all_top_indices = torch.cat(all_top_indices, dim=0)
        
        # Compute recall@K for each K
        recalls = {}
        for k in k_values:
            # Check if ground truth is in top-K for each query
            # gt_indices: [n_queries]
            # all_top_indices[:, :k]: [n_queries, k]
            hits = (all_top_indices[:, :k] == gt_indices.unsqueeze(1)).any(dim=1)
            recalls[k] = hits.float().mean().item()
        
        return recalls


def evaluate_skylines(index: TopicRetrievalIndex, k_values: list[int] = [1, 5, 10, 20]) -> dict:
    """
    Evaluate skyline baselines:
    1. Title -> Title: Using the exact article title as the query (upper bound)
    2. Random Label -> Title: Using a randomly selected synthetic label as query
    
    Args:
        index: Pre-built TopicRetrievalIndex
        k_values: K values for recall computation
        
    Returns:
        Dictionary with skyline results
    """
    n_topics = len(index.titles)
    ground_truth_indices = list(range(n_topics))
    
    results = {}
    
    # Skyline 1: Title -> Title
    print("\n=== Skyline 1: Title -> Title ===")
    print("(Upper bound: querying with the exact title)")
    title_recalls = index.compute_recall_at_k(
        query_texts=index.titles,
        ground_truth_indices=ground_truth_indices,
        k_values=k_values,
    )
    results["title_to_title"] = title_recalls
    for k, recall in title_recalls.items():
        print(f"  Recall@{k}: {recall:.4f}")
    
    # Skyline 2: Random Label -> Title
    print("\n=== Skyline 2: Random Synthetic Label -> Title ===")
    print("(Using one randomly selected synthetic label per topic)")
    
    # Select a random label for each topic (using first label for reproducibility)
    # In practice you might want to average over multiple random selections
    random_labels = [labels[0] for labels in index.labels]
    
    label_recalls = index.compute_recall_at_k(
        query_texts=random_labels,
        ground_truth_indices=ground_truth_indices,
        k_values=k_values,
    )
    results["random_label_to_title"] = label_recalls
    for k, recall in label_recalls.items():
        print(f"  Recall@{k}: {recall:.4f}")
    
    # Also evaluate with all 5 labels per topic
    print("\n=== Skyline 2b: All Synthetic Labels -> Title (macro avg) ===")
    print("(Using all 5 synthetic labels per topic)")
    
    all_labels = []
    all_gt_indices = []
    for i, labels in enumerate(index.labels):
        for label in labels:
            all_labels.append(label)
            all_gt_indices.append(i)
    
    all_label_recalls = index.compute_recall_at_k(
        query_texts=all_labels,
        ground_truth_indices=all_gt_indices,
        k_values=k_values,
    )
    results["all_labels_to_title"] = all_label_recalls
    for k, recall in all_label_recalls.items():
        print(f"  Recall@{k}: {recall:.4f}")
    
    return results


def analyze_failures(
    index: TopicRetrievalIndex,
    query_texts: list[str],
    ground_truth_indices: list[int],
    n_examples: int = 10,
) -> list[dict]:
    """
    Analyze failure cases where recall@1 fails.
    
    Returns examples with the query, correct topic, and what was retrieved instead.
    """
    top_indices, top_scores = index.query_batch(query_texts, k=5)
    
    failures = []
    for i, (query, gt_idx) in enumerate(zip(query_texts, ground_truth_indices)):
        predicted_idx = top_indices[i, 0].item()
        if predicted_idx != gt_idx:
            failures.append({
                "query": query,
                "correct_topic": index.titles[gt_idx],
                "predicted_topic": index.titles[predicted_idx],
                "predicted_score": top_scores[i, 0].item(),
                "top_5_topics": [index.titles[idx.item()] for idx in top_indices[i]],
                "top_5_scores": top_scores[i].tolist(),
            })
            
            if len(failures) >= n_examples:
                break
    
    return failures


def evaluate_custom_labels(
    index: TopicRetrievalIndex,
    labels: list[str],
    ground_truth_indices: Optional[list[int]] = None,
    k_values: list[int] = [1, 5, 10, 20, 50, 100],
) -> dict:
    """
    Evaluate a list of custom labels (e.g., from SelfIE or other methods).
    
    Args:
        index: Pre-built TopicRetrievalIndex
        labels: List of generated labels, one per topic (in order)
        ground_truth_indices: Optional custom ground truth indices.
                             If None, assumes labels[i] corresponds to topic i.
        k_values: K values for recall computation
        
    Returns:
        Dictionary with recalls and detailed results
    """
    if ground_truth_indices is None:
        ground_truth_indices = list(range(len(labels)))
    
    assert len(labels) == len(ground_truth_indices), \
        f"Number of labels ({len(labels)}) must match ground truth ({len(ground_truth_indices)})"
    
    # Compute recalls
    recalls = index.compute_recall_at_k(
        query_texts=labels,
        ground_truth_indices=ground_truth_indices,
        k_values=k_values,
    )
    
    # Get detailed per-label results
    max_k = max(k_values)
    top_indices, top_scores = index.query_batch(labels, k=max_k, show_progress=True)
    
    # We also need the similarity to the correct topic (not just top-k)
    # Re-embed labels and compute similarity to correct topics directly
    print("Computing similarity to correct topics...")
    query_embeddings = index._embed_texts(labels, show_progress=False)
    
    # Get correct topic embeddings
    gt_indices_tensor = torch.tensor(ground_truth_indices, device=index.device)
    correct_embeddings = index.title_embeddings[gt_indices_tensor]  # [n_labels, embed_dim]
    
    # Compute similarity to correct topic for each label
    correct_similarities = (query_embeddings * correct_embeddings).sum(dim=1)  # [n_labels]
    
    per_label_results = []
    reciprocal_ranks = []
    margins = []
    
    for i, (label, gt_idx) in enumerate(zip(labels, ground_truth_indices)):
        correct_topic = index.titles[gt_idx]
        predicted_idx = top_indices[i, 0].item()
        predicted_topic = index.titles[predicted_idx]
        predicted_score = top_scores[i, 0].item()
        correct_score = correct_similarities[i].item()
        
        # Find rank of correct topic (1-indexed)
        rank = None
        for r in range(max_k):
            if top_indices[i, r].item() == gt_idx:
                rank = r + 1
                break
        
        # Reciprocal rank (0 if not in top-k)
        rr = 1.0 / rank if rank is not None else 0.0
        reciprocal_ranks.append(rr)
        
        # Margin: correct_score - top1_score (positive if correct is top-1)
        margin = correct_score - predicted_score
        margins.append(margin)
        
        per_label_results.append({
            "label": label,
            "correct_topic": correct_topic,
            "predicted_topic": predicted_topic,
            "predicted_score": predicted_score,
            "correct_score": correct_score,
            "margin": margin,
            "correct_rank": rank,  # None if not in top max_k
            "reciprocal_rank": rr,
            "hit_at_1": predicted_idx == gt_idx,
        })
    
    # Compute aggregate metrics
    n = len(labels)
    mean_correct_similarity = sum(r["correct_score"] for r in per_label_results) / n
    mrr = sum(reciprocal_ranks) / n
    mean_margin = sum(margins) / n
    
    return {
        "recalls": recalls,
        "mean_correct_similarity": mean_correct_similarity,
        "mrr": mrr,  # Mean Reciprocal Rank
        "mean_margin": mean_margin,  # Avg (correct_score - top1_score)
        "per_label_results": per_label_results,
        "n_labels": n,
    }


def print_eval_summary(eval_results: dict, name: str = "Custom Labels") -> None:
    """Pretty-print evaluation results."""
    print(f"\n=== Evaluation: {name} ===")
    print(f"Total labels evaluated: {eval_results['n_labels']}")
    
    print("\nRecall@K (primary metric):")
    for k, recall in sorted(eval_results["recalls"].items()):
        print(f"  Recall@{k}: {recall:.4f} ({recall * 100:.2f}%)")
    
    print("\nAdditional metrics:")
    print(f"  MRR (Mean Reciprocal Rank): {eval_results['mrr']:.4f}")
    print(f"  Mean similarity to correct: {eval_results['mean_correct_similarity']:.4f}")
    print(f"  Mean margin (correct - top1): {eval_results['mean_margin']:.4f}")


def main():
    """Demo: Build index and evaluate skylines."""
    
    # Initialize with defaults
    config = TopicRetrievalConfig()
    index = TopicRetrievalIndex(config)
    
    # Build the index
    index.build_index()
    
    # Evaluate skylines
    k_values = [1, 5, 10, 20, 50, 100]
    results = evaluate_skylines(index, k_values=k_values)
    
    # Analyze some failure cases for the random label skyline
    print("\n=== Failure Case Analysis (Random Label -> Title) ===")
    random_labels = [labels[0] for labels in index.labels]
    ground_truth_indices = list(range(len(index.titles)))
    
    failures = analyze_failures(
        index,
        random_labels,
        ground_truth_indices,
        n_examples=5,
    )
    
    for i, f in enumerate(failures):
        print(f"\n--- Failure {i+1} ---")
        print(f"  Query: {f['query']}")
        print(f"  Correct: {f['correct_topic']}")
        print(f"  Predicted: {f['predicted_topic']} (score: {f['predicted_score']:.4f})")
        print(f"  Top 5: {f['top_5_topics']}")
    
    return results


if __name__ == "__main__":
    results = main()

