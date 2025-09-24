import torch

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

def log_tensor_stats(name, tensor):
    if tensor is None:
        print(f"{name}: None")
        return
    # Cast to float for statistics (won't affect original tensor)
    tensor_stats = tensor.float()
    stats = (
        f"{name}: shape={tuple(tensor.shape)}, "
        f"min={tensor.min().item():.3g}, "
        f"max={tensor.max().item():.3g}, "
        f"mean={tensor_stats.mean().item():.3g}, "
        f"std={tensor_stats.std().item():.3g}, "
    )
    print(stats)


def get_joint_valid_mask(batch):
    # Find all keys ending with '_valid'
    valid_keys = [k for k in batch.keys() if k.endswith('_valid') and isinstance(batch[k], torch.Tensor)]
    if not valid_keys:
        return None
    # Start with the first mask, then logically AND with others
    joint_mask = batch[valid_keys[0]].bool()
    for k in valid_keys[1:]:
        joint_mask = joint_mask & batch[k].bool()
    return joint_mask