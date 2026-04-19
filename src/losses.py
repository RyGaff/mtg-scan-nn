from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import BatchHardMiner

class BatchHardTripletLoss:
    def __init__(self, margin: float):
        self.loss_fn = TripletMarginLoss(margin=margin)
        self.miner = BatchHardMiner()

    def __call__(self, embeddings, labels):
        triplets = self.miner(embeddings, labels)
        return self.loss_fn(embeddings, labels, triplets)

def build_triplet_loss(margin: float):
    return BatchHardTripletLoss(margin=margin)
