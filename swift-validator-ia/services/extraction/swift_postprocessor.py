import re
from loguru import logger


class SwiftPostProcessor:
    """
    Clean and fix common OCR mistakes in SWIFT MT7xx messages.
    Minimal rules, focused on the most frequent field/tag errors.
    """

    TAG_AMOUNT_PATTERN = re.compile(r"(32B:)\s*(USD|EUR)\s*([0-9A-Z.,]+)", re.IGNORECASE)

    def clean(self, raw_text: str) -> str:
        if not raw_text:
            return ""

        text = raw_text

        # Normalize newlines -> spaces
        text = text.replace("\r", " ").replace("\n", " ")

        # Trim repeated whitespace
        text = re.sub(r"\s+", " ", text)

        # Fix misread SWIFT type
        text = text.replace("MT7OO", "MT700").replace("MT7Oo", "MT700")

        # Replace semi-colons with colons (common OCR confusion)
        text = text.replace(";", ":")

        # Fix 32B amounts by correcting OCR digits/letters
        def _fix_amount(match: re.Match) -> str:
            tag = match.group(1)
            currency = match.group(2).upper()
            amount = match.group(3)
            amount = (
                amount.replace("O", "0")
                .replace("S", "5")
                .replace("I", "1")
                .replace("B", "8")
            )
            amount = amount.replace(",", ".")
            return f"{tag} {currency} {amount}"

        text = self.TAG_AMOUNT_PATTERN.sub(_fix_amount, text)

        # Remove stray non printable characters except essential punctuation
        text = re.sub(r"[^\w\s:/{}.()-]", "", text)

        cleaned = text.strip()
        logger.debug(f"SwiftPostProcessor cleaned: {cleaned[:120]}...")
        return cleaned

