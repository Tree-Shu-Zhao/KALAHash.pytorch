from .csq import CsqCriterion
from .dch import DchCriterion
from .dsdh import DsdhCriterion
from .greedy_hash import GreedyHashCriterion
from .hashnet import HashNetCriterion
from .hswd import HSWDCriterion
from .kiddo import KiddoCriterion
from .mdsh import MdshCriterion
from .ortho import OrthoHashCriterion


def build_criterion(cfg, **kwargs):
    name = cfg.NAME.lower()
    if name == "kiddo":
        criterion = KiddoCriterion(cfg, kwargs["num_train"], kwargs["train_onehot_labels"])
    elif name == "dch":
        criterion = DchCriterion(cfg)
    elif name == "hashnet":
        criterion = HashNetCriterion(cfg)
    elif name == "dsdh":
        criterion = DsdhCriterion(cfg, kwargs["num_train"], kwargs["train_onehot_labels"])
    elif name == "greedy":
        criterion = GreedyHashCriterion(cfg)
    elif name == "csq":
        criterion = CsqCriterion(cfg)
    elif name == "ortho":
        criterion = OrthoHashCriterion(cfg)
    elif name == "mdsh":
        criterion = MdshCriterion(cfg, kwargs["num_train"], kwargs["train_onehot_labels"])
    elif name == "hswd":
        criterion = HSWDCriterion(cfg)
    else:
        raise ValueError(f"Can not find criterion name {cfg.NAME}!")

    return criterion
