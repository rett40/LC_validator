"""
parser_factory.py
Factory qui dispatch vers le bon parseur selon le type MT***.

Usage :
    factory  = SwiftParserFactory()
    entities = factory.parse(text, "MT700")
"""

from __future__ import annotations

from ..entities import SwiftEntities
from .MT700_parser import MT700Parser
from .MT720_parser import MT720Parser
from .MT760_parser import MT760Parser
from .MT790_parser import MT790Parser
from .MT799_parser import MT799Parser


class SwiftParserFactory:
    """
    Instancie et cache les parseurs MT***.
    Thread-safe (instances partagées, les parseurs sont stateless).
    """

    def __init__(self) -> None:
        self._parsers = {
            "MT700": MT700Parser(),
            "MT720": MT720Parser(),
            "MT760": MT760Parser(),
            "MT790": MT790Parser(),
            "MT799": MT799Parser(),
        }

    def parse(self, text: str, swift_type: str) -> SwiftEntities:
        """
        Dispatch vers le parseur du type détecté.
        Lève ValueError si le type n'est pas supporté.
        """
        parser = self._parsers.get(swift_type)
        if parser is None:
            raise ValueError(
                f"Type SWIFT non supporté : {swift_type!r}. "
                f"Supportés : {list(self._parsers)}"
            )
        return parser.parse(text)

    def supported_types(self) -> list[str]:
        return list(self._parsers.keys())