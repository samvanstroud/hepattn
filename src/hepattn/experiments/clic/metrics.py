import torch


class MaskInference:
    @staticmethod
    def basic_sigmoid(pred):
        """
        Assign hits to tracks if they have a high matching probability.
        Able to assign a hit to more than one track.
        """
        pred = pred.sigmoid() > 0.5
        return pred

    @staticmethod
    def basic_argmax(pred):
        """
        Assign hits to the track with the highest probability.
        Can only assign one hit to one track.
        """
        idx = pred.argmax(-2)
        pred = torch.full_like(pred, False).bool()
        pred[idx, torch.arange(len(idx))] = True
        return pred

    @staticmethod
    def weighted_argmax(pred, class_preds):
        """
        Assign hits to the track with the highest probabilithy, weighted with class pred confidence.
        Can only assign one hit to one track.
        This is used in the Maskformer paper.
        """
        idx = (pred.softmax(-2) * class_preds.max(-1)[0].unsqueeze(-1)).argmax(-2)
        pred = torch.zeros_like(pred).bool()
        pred[idx, torch.arange(len(idx))] = True
        return pred

    @staticmethod
    def exact_match(pred, tgt):
        """Perfect hit to track assignment"""
        if len(tgt) == 0:
            return torch.tensor(torch.nan)
        return (pred == tgt).all(-1).float().mean()

    @staticmethod
    def eff(pred, tgt):
        """Efficiency to assign correct hit to track"""
        return ((pred & tgt).sum(-1) / tgt.sum(-1)).mean()

    @staticmethod
    def pur(pred, tgt):
        """Purity of assigned hits on tracks"""
        return ((pred & tgt).sum(-1) / pred.sum(-1)).mean()
