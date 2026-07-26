"""Lazy model-family namespace for FastPLMs.

Model classes are resolved through Transformers AutoClasses and the typed
registry. Importing this package therefore does not load checkpoints, create
tokenizers, compile kernels, or initialize an accelerator runtime.
"""

from __future__ import annotations


__all__: tuple[str, ...] = ()
