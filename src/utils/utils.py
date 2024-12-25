import torch


def to_device(batch, device):
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    
    return batch