"""
bl_extractor
============
Extracteur spécialisé Bill of Lading — modèle LayoutLMv3 best_bl.

Usage :
  from release2.services.bl_extractor import BLExtractor

  ext = BLExtractor()
  result = ext.extract_from_pdf("doc.pdf", input_type="pdf_native")
  ext.print_results(result["fields"])
"""

from .bl_extraction_service import BLExtractor

__all__ = ["BLExtractor"]
