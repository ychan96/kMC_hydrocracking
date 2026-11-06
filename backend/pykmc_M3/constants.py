MAX_LENGTH = 30
C_LUMP     = MAX_LENGTH + 1          


def cn(n):
    """Return key infix for chain length n (n >= 1, capped at the lump key)."""
    if n < 1:
        raise ValueError(f"chain length must be >= 1, got {n}")
    return f'c{n}' if n <= MAX_LENGTH else f'c{C_LUMP}plus'