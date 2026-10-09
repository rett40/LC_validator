"""Extraction d'entités nommées (NER) pour messages SWIFT."""

from .swift_ner_extractor import (
    SwiftNERExtractor,
    NERResult,
    NamedEntity,
    EntityRelation,
)

__all__ = [
    "SwiftNERExtractor",
    "NERResult",
    "NamedEntity",
    "EntityRelation",
]
