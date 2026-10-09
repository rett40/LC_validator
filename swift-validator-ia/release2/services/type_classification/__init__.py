"""
type_classification
===================
Wrapper unifié de classification de documents LC.

Usage :
  from release2.services.type_classification import TypeClassifier

  clf = TypeClassifier()
  doc_type, confidence, probas = clf.classify("doc.pdf")
"""

from .classifier import TypeClassifier

__all__ = ["TypeClassifier"]
