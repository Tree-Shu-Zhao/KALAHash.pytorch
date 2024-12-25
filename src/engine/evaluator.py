import torch
from tqdm import tqdm

from src.utils import to_device


class HashingEvaluator():
    def __init__(self, bits, topk, device):
        self.bits = bits
        self.topk = topk
        self.device = device

    def evaluate(self, model, query_dataloader, gallery_dataloader):
        model.set_model_mode("eval")

        query_codes = self._generate_hash_codes(model, query_dataloader)
        query_labels = query_dataloader.dataset.onehot_labels.float().to(self.device)
        gallery_codes = self._generate_hash_codes(model, gallery_dataloader)
        gallery_labels = gallery_dataloader.dataset.onehot_labels.float().to(self.device)

        mAP = self._calculate_map(query_codes, query_labels, gallery_codes, gallery_labels, self.topk)
        P, R = self._calculate_pr_curve(query_codes, query_labels, gallery_codes, gallery_labels)

        model.set_model_mode("train")

        return {
            "mAP": mAP, 
            "precision": P, 
            "recall": R,
        }

    def _generate_hash_codes(self, model, dataloader):
        hash_codes = torch.zeros(len(dataloader.dataset), self.bits).to(self.device)
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating hash codes"):
                batch = to_device(batch, self.device)
                indices = batch["indices"]

                outputs = model(batch)
                hash_codes[indices, :] = outputs["image_hash_features"].sign().float()
        
        return hash_codes
    
    def _calculate_map(self, query_codes, query_labels, gallery_codes, gallery_labels, topk=None):
        num_query = query_labels.shape[0]
        mean_AP = 0.0

        for i in tqdm(range(num_query), "Evaluating mAP"):
            # Retrieve images from database
            gallery = (query_labels[i, :] @ gallery_labels.t() > 0).float()

            # Calculate hamming distance
            hamming_dist = 0.5 * (gallery_codes.shape[1] - query_codes[i, :] @ gallery_codes.t())

            # Arrange position according to hamming distance
            if topk is not None:
                gallery = gallery[torch.argsort(hamming_dist)][:topk]
            else:
                gallery = gallery[torch.argsort(hamming_dist)]

            # Retrieval count
            gallery_cnt = gallery.sum().int().item()

            # Cannot retrieve images
            if gallery_cnt == 0:
                continue

            # Generate score for every position
            score = torch.linspace(1, gallery_cnt, gallery_cnt, device=query_codes.device)

            # Acquire index
            index = (torch.nonzero(gallery == 1).squeeze() + 1.0).float()

            mean_AP += (score / index).mean()

        mean_AP = mean_AP / num_query
        return mean_AP.item()
    
    def _calculate_pr_curve(
        self,
        query_codes, 
        query_targets, 
        gallery_codes, 
        gallery_targets, 
    ):
        num_query = query_codes.shape[0]
        P = []
        R = []

        for recall in tqdm(torch.linspace(0.01, 1, 15), desc="PR curve"):
            recall = recall.item()
            precision = 0
            for i in range(num_query):
                # Retrieve images from database
                gallery = (query_targets[i, :] @ gallery_targets.t() > 0).float()

                # Calculate hamming distance
                hamming_dist = 0.5 * (gallery_codes.shape[1] - query_codes[i, :] @ gallery_codes.t())

                # Arrange position according to hamming distance
                gallery = gallery[torch.argsort(hamming_dist)]

                if not gallery.sum().item():
                    continue

                gallery_index = torch.nonzero(gallery)
                recall_position = gallery_index[int(gallery.sum().item() * recall) - 1]

                gallery_recall = gallery[:max(1, recall_position)]
                precision += gallery_recall.sum().item() / len(gallery_recall)
            precision /= num_query

            P.append(precision)
            R.append(recall)

        return P, R
