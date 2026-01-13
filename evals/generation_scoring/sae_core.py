#!/usr/bin/env python3
"""
Core SAE and language model components for SAE evaluations.
Re-exports from the shared selfie_adapters package.
"""

import sys
from pathlib import Path

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from selfie_adapters.sae_utils import load_sae, ObservableLanguageModel

__all__ = ['load_sae', 'ObservableLanguageModel']
