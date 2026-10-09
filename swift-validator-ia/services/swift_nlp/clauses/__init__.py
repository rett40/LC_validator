"""Classification de clauses SWIFT."""

from .swift_clause_classifier import (
    SwiftClauseClassifier,
    ClauseClassification,
    train_clause_classifier,
)

__all__ = [
    "SwiftClauseClassifier",
    "ClauseClassification",
    "train_clause_classifier",
]
