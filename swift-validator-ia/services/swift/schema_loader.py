"""
schema_loader.py
Charge dynamiquement les schémas MT*** au format JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


class SwiftSchemaLoader:
    """
    Charge les schémas depuis data/swift_schema/mt*_schema.json.
    """

    def __init__(self,
                 base_dir: Path | None = None,
                 schema_dir: Path | None = None) -> None:
        self.base_dir   = base_dir  or Path(__file__).resolve().parents[2]
        self.schema_dir = schema_dir
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self._load_all()

    def _load_all(self) -> None:
        candidates = []
        if self.schema_dir:
            candidates.append(self.schema_dir)
        candidates.append(self.base_dir / "data" / "swift_schema")
        candidates.append(self.base_dir)

        schema_dir = next((c for c in candidates if c.exists()), None)
        if schema_dir is None:
            return

        for path in sorted(schema_dir.glob("mt*_schema.json")):
            try:
                with path.open(encoding="utf-8") as f:
                    data = json.load(f)
                mt = data.get("message_type")
                if mt:
                    self.schemas[mt] = data
            except Exception:
                continue  # schéma corrompu ignoré silencieusement

    def get(self, message_type: str) -> Dict[str, Any] | None:
        return self.schemas.get(message_type)

    def all_types(self) -> list[str]:
        return list(self.schemas.keys())