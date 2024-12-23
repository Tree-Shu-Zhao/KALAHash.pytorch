from loguru import logger


def reset_cfg(cfg, params):
    # Optimizer
    if "LR" in params.keys():
        cfg.train.optimizer.LR = params["LR"]
        logger.info(f"Reset LR to {params['LR']}")
    if "MOMENTUM" in params.keys():
        cfg.train.optimizer.MOMENTUM = params["MOMENTUM"]
        logger.info(f"Reset MOMENTUM to {params['MOMENTUM']}")
    if "WEIGHT_DECAY" in params.keys():
        cfg.train.optimizer.WEIGHT_DECAY = params["WEIGHT_DECAY"]
        logger.info(f"Reset WEIGHT_DECAY to {params['WEIGHT_DECAY']}")
    if "BATCH_SIZE" in params.keys():
        cfg.train.BATCH_SIZE = params["BATCH_SIZE"]
        logger.info(f"Reset BATCH_SIZE to {params['BATCH_SIZE']}")
    if "EPOCHS" in params.keys():
        cfg.train.EPOCHS = params["EPOCHS"]
        logger.info(f"Reset EPOCHS to {params['EPOCHS']}")
    
    # Model
    if "NCTX" in params.keys():
        cfg.model.prompt_learner.N_CTX = params["NCTX"]
        logger.info(f"Reset NCTX to {params['NCTX']}")
    if "GENERATOR_NAME" in params.keys():
        cfg.model.lora.GENERATOR_NAME = params["GENERATOR_NAME"]
        logger.info(f"Reset GENERATOR_NAME to {params['GENERATOR_NAME']}")
    if "LORA_START_LAYERS" in params.keys():
        cfg.model.lora.LORA_START_LAYERS = params["LORA_START_LAYERS"]
        logger.info(f"Reset LORA_START_LAYERS to {params['LORA_START_LAYERS']}")
    if "LORA_POSITION" in params.keys():
        cfg.model.lora.LORA_POSITION = params["LORA_POSITION"]
        logger.info(f"Reset LORA_POSITION to {params['LORA_POSITION']}")
    if "RANK" in params.keys():
        cfg.model.lora.RANK = params["RANK"]
        logger.info(f"Reset RANK to {params['RANK']}")
    if "LORA_STRENGTH" in params.keys():
        cfg.model.lora.LORA_STRENGTH = params["LORA_STRENGTH"]
        logger.info(f"Reset LORA_STRENGTH to {params['LORA_STRENGTH']}")
    if "DROPOUT_RATE" in params.keys():
        cfg.model.lora.DROPOUT_RATE = params["DROPOUT_RATE"]
        logger.info(f"Reset DROPOUT_RATE to {params['DROPOUT_RATE']}")
    if "HIDDEN_DIM" in params.keys():
        cfg.model.lora.HIDDEN_DIM = params["HIDDEN_DIM"]
        logger.info(f"Reset HIDDEN_DIM to {params['HIDDEN_DIM']}")

    # Criterion
    if "TEMPERATURE" in params.keys():
        cfg.criterion.TEMPERATURE = params["TEMPERATURE"]
        logger.info(f"Reset TEMPERATURE to {params['TEMPERATURE']}")
    if "ALPHA" in params.keys():
        cfg.criterion.ALPHA = params["ALPHA"]
        logger.info(f"Reset ALPHA to {params['ALPHA']}")
    if "BETA" in params.keys():
        cfg.criterion.BETA = params["BETA"]
        logger.info(f"Reset BETA to {params['BETA']}")
    if "MU" in params.keys():
        cfg.criterion.MU = params["MU"]
        logger.info(f"Reset MU to {params['MU']}")
    if "NU" in params.keys():
        cfg.criterion.NU = params["NU"]
        logger.info(f"Reset NU to {params['NU']}")
    if "ETA" in params.keys():
        cfg.criterion.ETA = params["ETA"]
        logger.info(f"Reset ETA to {params['ETA']}")
    if "NUM_DCC" in params.keys():
        cfg.criterion.NUM_DCC = params["NUM_DCC"]
        logger.info(f"Reset NUM_DCC to {params['NUM_DCC']}")
    if "TAU" in params.keys():
        cfg.criterion.classification_criterion.TAU = params["TAU"]
        logger.info(f"Reset TAU to {params['TAU']}")
    if "GAMMA" in params.keys():
        cfg.criterion.GAMMA = params["GAMMA"]
        logger.info(f"Reset GAMMA to {params['GAMMA']}")
    if "HASHNET_ALPHA" in params.keys():
        cfg.criterion.similarity_criterion.HASHNET_ALPHA = params["HASHNET_ALPHA"]
        logger.info(f"Reset HASHNET_ALPHA to {params['HASHNET_ALPHA']}")
    if "HASHNET_STEP" in params.keys():
        cfg.criterion.similarity_criterion.HASHNET_STEP = params["HASHNET_STEP"]
        logger.info(f"Reset HASHNET_STEP to {params['HASHNET_STEP']}")
    if "DSDH_MU" in params.keys():
        cfg.criterion.DSDH_MU = params["DSDH_MU"]
        logger.info(f"Reset DSDH_MU to {params['DSDH_MU']}")
    if "DSDH_NU" in params.keys():
        cfg.criterion.DSDH_NU = params["DSDH_NU"]
        logger.info(f"Reset DSDH_NU to {params['DSDH_NU']}")
    if "DSDH_ETA" in params.keys():
        cfg.criterion.DSDH_ETA = params["DSDH_ETA"]
        logger.info(f"Reset DSDH_ETA to {params['DSDH_ETA']}")
    if "GREEDY_ALPHA" in params.keys():
        cfg.criterion.quantization_criterion.ALPHA = params["GREEDY_ALPHA"]
        logger.info(f"Reset GREEDY_ALPHA to {params['GREEDY_ALPHA']}")
    if "CSQ_LAMBDA" in params.keys():
        cfg.criterion.LAMBDA = params["CSQ_LAMBDA"]
        logger.info(f"Reset CSQ_LAMBDA to {params['CSQ_LAMBDA']}")
    if "ORTHO_SCALE" in params.keys():
        cfg.criterion.similarity_criterion.SCALE = params["ORTHO_SCALE"]
        logger.info(f"Reset ORTHO_SCALE to {params['ORTHO_SCALE']}")
    if "ORTHO_MARGIN" in params.keys():
        cfg.criterion.similarity_criterion.MARGIN = params["ORTHO_MARGIN"]
        logger.info(f"Reset ORTHO_MARGIN to {params['ORTHO_MARGIN']}")

    return cfg
