import unittest
from deepgo.metrics import compute_roc, compute_metrics
import numpy as np


class TestMetrics(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        pass

    def test_compute_roc(self):
        labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        preds = np.array([0.1, 0.6, 0.7, 0.8, 1, 0.1, 0.2, 0.1, 0.1, 0.2])
        roc_auc = compute_roc(labels, preds)
        self.assertEqual(roc_auc, 0.8600000000000001)

    
