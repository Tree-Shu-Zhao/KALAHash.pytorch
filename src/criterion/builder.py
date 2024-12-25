from .kiddo import KiddoCriterion


def build_criterion(cfg, **kwargs):
    name = cfg.NAME.lower()
    if name == "kiddo":
        criterion = KiddoCriterion(cfg, kwargs["num_train"], kwargs["train_onehot_labels"])
    else:
        raise ValueError(f"Can not find criterion name {cfg.NAME}!")

    return criterion
