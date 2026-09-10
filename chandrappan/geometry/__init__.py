"""Geometric registration verification utilities."""

from .coverage import spatial_coverage
from .verification import VerificationConfig, VerificationResult, verify_registration

__all__ = ["VerificationConfig", "VerificationResult", "spatial_coverage", "verify_registration"]
