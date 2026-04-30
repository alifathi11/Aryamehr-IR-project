import math
from typing import List

class Evaluation:
    def __init__(self, name: str):
        self.name = name

    def _validate(self, actual: List[List[str]], predicted: List[List[str]]):
        if len(actual) != len(predicted):
            raise ValueError("actual and predicted must have the same length")

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates macro precision.
        """
        precisions = []

        for act, pred in zip(actual, predicted):
            act_set = set(act)
            pred_set = set(pred)

            if len(pred_set) == 0:
                precisions.append(0.0)
                continue

            intersection = act_set & pred_set

            precision = len(intersection) / len(pred_set)
            precisions.append(precision)

        if precisions:
            return sum(precisions) / len(precisions)
        else:
            return 0.0

    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates macro recall.
        """
        recalls = []

        for act, pred in zip(actual, predicted): 
            act_set = set(act)
            pred_set = set(pred)

            if len(act_set) == 0:
                recalls.append(0.0)
                continue 

            intersection = act_set & pred_set 

            recall = len(intersection) / len(act_set)
            recalls.append(recall)

        if recalls:
            return sum(recalls) / len(recalls)
        else:
            return 0.0

    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates F1 score.
        """
        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)

        if precision + recall == 0:
            return 0.0
        else:
            return (2 * precision * recall) / (precision + recall)

    def _average_precision_single(self, actual: List[str], predicted: List[str]) -> float:

        actual_set = set(actual)

        if not actual_set:
            return 0.0

        hits = 0
        precisions = []

        for k, doc_id in enumerate(predicted, start=1):
            if doc_id in actual_set:
                hits += 1
                precisions.append(hits / k)

        return sum(precisions) / len(actual_set)

    def calculate_AP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates mean AP across all queries.
        """
        average_precisions = []

        for act, pred in zip(actual, predicted):
            average_precision = self._average_precision_single(act, pred)
            average_precisions.append(average_precision)

        if average_precisions:
            return sum(average_precisions) / len(average_precisions)
        else: 
            return 0.0

    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates MAP.
        """
        return self.calculate_AP(actual, predicted)

    def _dcg_single(self, actual: List[str], predicted: List[str]) -> float:
        
        actual_set = set(actual)

        dcg = 0.0

        for i, doc_id in enumerate(predicted, start=1):
            if doc_id in actual_set:
                dcg += 1 / math.log2(i + 1)

        return dcg 

    def calculate_DCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates mean DCG.
        """
        dcgs = []

        for act, pred in zip(actual, predicted):
            dcgs.append(self._dcg_single(act, pred))

        if dcgs:
            return sum(dcgs) / len(dcgs)
        else: 
            return 0.0
        
    def _ndcg_single(self, actual: List[str], predicted: List[str]) -> float:
        
        dcg = self._dcg_single(actual, predicted)

        actual_set = set(actual)
        ideal_rels = min(len(actual_set), len(predicted))

        idcg = 0.0
        for i in range(1, ideal_rels + 1):
            idcg += 1 / math.log2(i + 1)

        if idcg == 0:
            return 0.0 
        
        return dcg / idcg 

    def calculate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates mean NDCG.
        """
        ndcgs = []

        for act, pred in zip(actual, predicted):
            ndcgs.append(self._ndcg_single(act, pred))

        if ndcgs: 
            return sum(ndcgs) / len(ndcgs)
        else:
            return 0.0

    def _rr_single(self, actual: List[str], predicted: List[str]) -> float: 

        actual_set = set(actual)

        for i, doc_id in enumerate(predicted, start=1):
            if doc_id in actual_set:
                return 1.0 / i
            
        return 0.0

    def calculate_RR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculate reciprocal rank.
        """
        rrs = []

        for act, pred in zip(actual, predicted):
            rrs.append(self._rr_single(act, pred))

        if rrs:
            return sum(rrs) / len(rrs)
        else: 
            return 0.0

    def calculate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates MRR.
        """ 
        return self.calculate_RR(actual, predicted)
        
    def print_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Prints the evaluation metrics.
        """
        print(f"name = {self.name}")
        print(f"Precision = {precision:.6f}")
        print(f"Recall = {recall:.6f}")
        print(f"F1 = {f1:.6f}")
        print(f"AP = {ap:.6f}")
        print(f"MAP = {map:.6f}")
        print(f"DCG = {dcg:.6f}")
        print(f"NDCG = {ndcg:.6f}")
        print(f"RR = {rr:.6f}")
        print(f"MRR = {mrr:.6f}")

    def log_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Use Wandb to log the evaluation metrics.
        """
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'ap': ap,
                    'map': map,
                    'dcg': dcg,
                    'ndcg': ndcg,
                    'rr': rr,
                    'mrr': mrr,
                })
        except Exception:
            pass

    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        Call all functions to calculate evaluation metrics.
        """
        self._validate(actual, predicted)

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)

        ap = self.calculate_AP(actual, predicted)
        map = self.calculate_map(actual, predicted)

        dcg = self.calculate_DCG(actual, predicted)
        ndcg = self.calculate_NDCG(actual, predicted)

        rr = self.calculate_RR(actual, predicted)
        mrr = self.calculate_MRR(actual, predicted)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap": ap,
            "map": map,
            "dcg": dcg,
            "ndcg": ndcg,
            "rr": rr,
            "mrr": mrr,
        }


