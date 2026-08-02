def window(values, size):
    if size <= 0:
        raise ValueError("size must be positive")
    if size > len(values):
        return []
    return [tuple(values[i:i + size]) for i in range(len(values) - size + 1)]
