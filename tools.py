

def update_config(cfg, keys, value):
    """Recursively update nested dictionary cfg with keys list and set to value."""
    d = cfg
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    # Try to interpret value as Python literal
    try:
        import ast
        value = ast.literal_eval(value)
    except Exception:
        pass
    d[keys[-1]] = value