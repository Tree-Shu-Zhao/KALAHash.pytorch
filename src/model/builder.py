from .kala_hash import KalaHash


def build_model(cfg, classnames):
    name = cfg.NAME.lower()
    if name == "kalahash":
        model = KalaHash(cfg, classnames)
    else:
        raise ValueError(f"Model {name} not found")
    return model