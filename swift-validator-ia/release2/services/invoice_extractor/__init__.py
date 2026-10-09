"""
invoice_extractor
=================
Extracteur spécialisé Commercial Invoice — modèle LayoutLMv3 best_invoice.

Usage :
  from release2.services.invoice_extractor import InvoiceExtractor

  ext = InvoiceExtractor()
  result = ext.extract_from_pdf("invoice.pdf", input_type="pdf_native")
  ext.print_results(result["fields"])
"""

from .invoice_extraction_service import InvoiceExtractor

__all__ = ["InvoiceExtractor"]
