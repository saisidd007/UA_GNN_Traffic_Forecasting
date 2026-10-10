import hashlib
import pickle
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkGraphTests(unittest.TestCase):
    def test_official_metr_graph_matches_sensors_and_provenance(self):
        folder = ROOT / "METR-LA/data/metr-la"
        path = folder / "adj_mx.pkl"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(digest, "a35687c6e15aa228dc45027b0ed2a0ea0f4ec78f573deb992c595092d12f61b3")
        sensors = pd.read_csv(folder / "METR-LA.csv", nrows=0).columns[1:].astype(str).tolist()
        with path.open("rb") as stream:
            graph_ids, _, adjacency = pickle.load(stream, encoding="latin1")
        self.assertEqual(graph_ids, sensors)
        self.assertEqual(adjacency.shape, (207, 207))
        self.assertEqual(int((adjacency > .1).sum()-np.trace(adjacency > .1)), 1515)

    def test_pems_graph_matches_sensors(self):
        folder = ROOT / "PEMS-BAY/data/pems-bay"
        sensors = pd.read_csv(folder / "PEMS-BAY.csv", nrows=0).columns[1:].astype(str).tolist()
        with (folder / "adj_mx_bay.pkl").open("rb") as stream:
            graph_ids, _, adjacency = pickle.load(stream, encoding="latin1")
        self.assertEqual(graph_ids, sensors)
        self.assertEqual(adjacency.shape, (325, 325))
        self.assertEqual(int((adjacency > .1).sum()-np.trace(adjacency > .1)), 2369)


if __name__ == "__main__":
    unittest.main()
