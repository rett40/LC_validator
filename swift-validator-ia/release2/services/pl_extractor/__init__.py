"""
pl_extractor
============
Extracteur spécialisé Packing List — modèle LayoutLMv3 best_pl.

Usage :
  from release2.services.pl_extractor import PLExtractor

  ext = PLExtractor()
  result = ext.extract_from_pdf("packing_list.pdf", input_type="pdf_native")
  ext.print_results(result["fields"])
"""

from .pl_extraction_service import PLExtractor

__all__ = ["PLExtractor"]
