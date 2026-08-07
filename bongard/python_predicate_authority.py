"""Stable identifier for the canonical pure-Python predicate authority.

Keep this leaf module dependency-free.  Active pipelines may share the
authority identifier without importing an older evaluator, vision stack, Lean
bridge, or benchmark runner merely to name the decision authority.
"""

PYTHON_PREDICATE_AUTHORITY_ID = (
    "bongard.grounded-multimodal-predicate-authority/python-v1"
)


__all__ = ("PYTHON_PREDICATE_AUTHORITY_ID",)
