# set root directory
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
from loguru import logger
from omegaconf import OmegaConf

from src.dataset import build_dataloaders
from src.engine import Trainer
from src.model import build_model
from src.utils import init_env


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg):
    # Setup logger, code backup, hyper-parameteres records, etc...
    init_env(cfg)

    # Print configuration
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    # Build dataloaders and model
    dataloaders = build_dataloaders(cfg.dataset)
    model = build_model(cfg.model, dataloaders["train"].dataset.CLASS_NAMES)

    # Build trainer
    trainer = Trainer(cfg, dataloaders, model)

    if cfg.test.EVAL_ONLY:
        logger.info("!!!!!!!EVALUATION ONLY!!!!!!!")
        result = trainer.evaluate(checkpoint_path=cfg.test.CHECKPOINT_PATH)
        mAP = result["mAP"]
        logger.info("Final mAP: {:.4f}".format(mAP))
        return mAP

    trainer.train()

    if trainer.best_map == 0:
        result = trainer.evaluate()
        mAP = result["mAP"]
    else:
        mAP = trainer.best_map
    
    logger.info("Final mAP: {:.4f}".format(mAP))


if __name__ == "__main__":
    main()
