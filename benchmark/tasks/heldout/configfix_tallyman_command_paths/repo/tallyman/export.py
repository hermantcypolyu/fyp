from typing import Dict


def format_totals(summed: Dict[str, float], *, width: int = 12) -> str:
    """One `category  amount` line per entry, amounts right-aligned."""
    lines = []
    for category, amount in summed.items():
        lines.append(f"{category.ljust(width)}{amount:>10.2f}")
    return "\n".join(lines)


def format_summary(summed: Dict[str, float], total: float, leader: str) -> str:
    """The per-category block, then a total line, then the leading category."""
    body = format_totals(summed)
    parts = [body] if body else []
    parts.append(f"{'total'.ljust(12)}{total:>10.2f}")
    parts.append(f"top: {leader}")
    return "\n".join(parts)
