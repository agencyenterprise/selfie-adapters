#!/usr/bin/env python3
"""Lightweight inference utilities for trained SelfIE adapters."""

from typing import Optional, Dict, Any
import torch

from selfie_adapters.projection import create_projection_module


class SelfIEAdapter:
    """
    Lightweight loader for inference with trained SelfIE adapters.
    
    This class allows you to load a trained projection module from a checkpoint
    and use it for inference without loading the full training infrastructure
    (language model, optimizer, etc.).
    
    Example:
        >>> adapter = SelfIEAdapter("checkpoint.pt")
        >>> soft_tokens = adapter.transform(sae_vectors)
        >>> print(adapter.get_metadata())
    """
    
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        """
        Load a trained SelfIE adapter from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint .pt file
            device: Device to load projection on (e.g., "cpu", "cuda", "cuda:0").
                   If None, uses "cuda" if available, otherwise "cpu".
        
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint format is invalid or unsupported
        """
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Load checkpoint
        # Note: weights_only=False is required because checkpoint contains config dict
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Validate checkpoint format
        if "projection_state" not in checkpoint:
            raise ValueError(
                f"Invalid checkpoint format: missing 'projection_state'. "
                f"Available keys: {list(checkpoint.keys())}"
            )
        
        if "config" not in checkpoint:
            raise ValueError(
                f"Invalid checkpoint format: missing 'config'. "
                f"This checkpoint may be too old or corrupted."
            )
        
        # Extract config
        config_dict = checkpoint["config"]
        if "projection" not in config_dict:
            raise ValueError("Invalid checkpoint: config missing 'projection' section")
        
        proj_config = config_dict["projection"]
        
        # Get model dimension
        if "model_dim" in checkpoint:
            model_dim = checkpoint["model_dim"]
        else:
            # Old checkpoint format - infer from state dict
            model_dim = self._infer_dim_from_state(
                checkpoint["projection_state"],
                proj_config
            )
        
        # Store metadata
        self.model_dim = model_dim
        self.config = proj_config
        self.checkpoint_path = checkpoint_path
        self.checkpoint_format_version = checkpoint.get("checkpoint_format_version", 0)
        self.global_step = checkpoint.get("global_step", None)
        self.best_val_loss = checkpoint.get("best_val_loss", None)
        
        # Recreate projection module with exact training configuration
        print(f"Loading {proj_config['type']} projection (dim={model_dim}) from {checkpoint_path}")
        self.projection = create_projection_module(
            projection_type=proj_config["type"],
            dim=model_dim,
            normalize_input=proj_config["normalize_input"],
            device=self.device,
            init_scale=proj_config.get("init_scale", 30.0),
            low_rank_rank=proj_config.get("low_rank_rank"),
            low_rank_init_factor=proj_config.get("low_rank_init_factor"),
        )
        
        # Load trained weights
        self.projection.load_state_dict(checkpoint["projection_state"])
        
        # Set to evaluation mode
        self.projection.eval()
        
        # Verify parameter count if available
        if "projection_num_params" in checkpoint:
            expected = checkpoint["projection_num_params"]
            actual = self.projection.num_parameters()
            if expected != actual:
                print(f"WARNING: Parameter count mismatch. Expected {expected}, got {actual}")
        
        print(f"✓ Loaded adapter with {self.projection.num_parameters():,} parameters")
    
    def _infer_dim_from_state(self, state_dict: Dict[str, torch.Tensor], config: Dict[str, Any]) -> int:
        """Infer model dimension from state dict (fallback for old checkpoints)."""
        ptype = config["type"]
        
        # Try common parameters
        if "bias" in state_dict:
            return state_dict["bias"].shape[0]
        
        if "scale_direction" in state_dict:
            return state_dict["scale_direction"].shape[0]
        
        # Architecture-specific inference
        if ptype == "full_rank":
            if "weight" in state_dict:
                return state_dict["weight"].shape[0]
        
        elif ptype == "scalar_affine_plus_low_rank":
            if "U" in state_dict:
                return state_dict["U"].shape[0]
        
        elif ptype == "scale_only":
            raise ValueError(
                "Cannot infer model_dim from scale_only projection without additional info."
            )
        
        raise ValueError(
            f"Cannot infer model_dim from {ptype} checkpoint. "
            f"Available state keys: {list(state_dict.keys())}."
        )
    
    @torch.no_grad()
    def transform(self, vectors: torch.Tensor, normalize_input: Optional[bool] = None) -> torch.Tensor:
        """
        Apply trained projection to input vectors.
        
        Args:
            vectors: Input vectors of shape (batch_size, model_dim) or (model_dim,)
            normalize_input: Override normalization behavior:
                - None (default): Use training configuration
                - True: Force L2 normalization of inputs
                - False: Skip normalization
        
        Returns:
            Soft token embeddings of shape (batch_size, model_dim) or (model_dim,)
        """
        # Handle single vector case
        single_vector = vectors.ndim == 1
        if single_vector:
            vectors = vectors.unsqueeze(0)
        
        # Validate shape
        if vectors.shape[-1] != self.model_dim:
            raise ValueError(
                f"Input vectors have dimension {vectors.shape[-1]}, "
                f"but adapter expects {self.model_dim}"
            )
        
        # Move to device and transform
        vectors = vectors.to(self.device)
        vectors_f32 = vectors.float()
        
        # Handle normalization override
        if normalize_input is None:
            soft_tokens = self.projection(vectors_f32)
        else:
            original_normalize = self.projection.normalize_input
            self.projection.normalize_input = normalize_input
            try:
                soft_tokens = self.projection(vectors_f32)
            finally:
                self.projection.normalize_input = original_normalize
        
        # Convert back to input dtype
        soft_tokens = soft_tokens.to(vectors.dtype)
        
        if single_vector:
            soft_tokens = soft_tokens.squeeze(0)
        
        return soft_tokens
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the loaded adapter.
        
        Returns:
            Dictionary with architecture and training information
        """
        metadata = {
            "projection_type": self.config["type"],
            "model_dim": self.model_dim,
            "normalize_input": self.config["normalize_input"],
            "num_parameters": self.projection.num_parameters(),
            "device": str(self.device),
            "checkpoint_format_version": self.checkpoint_format_version,
        }
        
        if self.global_step is not None:
            metadata["global_step"] = self.global_step
        
        if self.best_val_loss is not None:
            metadata["best_val_loss"] = self.best_val_loss
        
        for key in ["init_scale", "low_rank_rank", "low_rank_init_factor"]:
            if key in self.config and self.config[key] is not None:
                metadata[key] = self.config[key]
        
        return metadata
    
    def get_projection_stats(self) -> Dict[str, float]:
        """Get statistics about the projection parameters (scale, bias norm, etc.)."""
        stats = {}
        
        if hasattr(self.projection, "get_scale"):
            stats["scale"] = self.projection.get_scale()
        
        if hasattr(self.projection, "get_bias_norm"):
            stats["bias_norm"] = self.projection.get_bias_norm()
        
        if hasattr(self.projection, "get_weight_norm"):
            stats["weight_norm"] = self.projection.get_weight_norm()
        
        if hasattr(self.projection, "get_low_rank_norm"):
            stats["low_rank_norm"] = self.projection.get_low_rank_norm()
        
        return stats
    
    def __repr__(self) -> str:
        return (
            f"SelfIEAdapter("
            f"type={self.config['type']}, "
            f"dim={self.model_dim}, "
            f"params={self.projection.num_parameters():,}, "
            f"device={self.device})"
        )


def load_adapter(checkpoint_path: str, device: Optional[str] = None) -> SelfIEAdapter:
    """
    Convenience function to load a SelfIE adapter.
    
    Args:
        checkpoint_path: Path to checkpoint .pt file
        device: Device to load on (default: auto-detect)
    
    Returns:
        Loaded SelfIEAdapter instance
    
    Example:
        >>> adapter = load_adapter("checkpoint.pt")
        >>> soft_tokens = adapter.transform(vectors)
    """
    return SelfIEAdapter(checkpoint_path, device)
