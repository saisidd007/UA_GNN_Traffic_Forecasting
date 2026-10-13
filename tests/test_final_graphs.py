import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from corrected_experiment import prepare
from paper_model import PaperUAGNN
from generate_final_graphs import generate


class FinalGraphTests(unittest.TestCase):
    def test_all_pdf_figures_and_target_time_alignment(self):
        torch.set_num_threads(1)
        timestamps = pd.date_range('2020-01-01', periods=4000, freq='5min').astype(str).tolist()
        raw = np.random.default_rng(5).uniform(10, 60, (4000, 3)).astype(np.float32)
        values, observed, _, _, splits, _ = prepare(raw, val_ratio=.1)
        starts = splits['test']
        targets = np.stack([values[i+12:i+24].T for i in starts])
        masks = np.stack([observed[i+12:i+24].T for i in starts])
        graph = np.ones((3, 3), dtype=np.float32)-np.eye(3, dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            config = dict(dataset='METR-LA', seed=42, epochs_completed=50, best_epoch=45,
                          architecture='paper_interpretation_v1', train_ratio=.7, val_ratio=.1,
                          hidden=8, no_dropout=False, no_diffusion=False, no_aleatoric=False,
                          graph_threshold=.1)
            (folder/'metrics.json').write_text(json.dumps(config))
            (folder/'training.json').write_text(json.dumps([
                dict(epoch=i, train_loss=1/i, val_mse=1/i) for i in range(1,51)]))
            np.savez(folder/'predictions.npz', mean=targets+.1, target=targets,
                     epistemic=np.full_like(targets,.1), aleatoric=np.full_like(targets,.2), target_observed=masks)
            torch.save(PaperUAGNN(hidden=8).state_dict(), folder/'best.pt')
            args = argparse.Namespace(correlation_sensors=3, sensor_index=0, horizon_step=12,
                day=None, reuse_robustness=False, robustness_windows=4, device='cpu',
                batch_size=2, mc_passes=2, corruption_seeds=[17,29], failure_rates=[.1,.5])
            with patch('generate_final_graphs.load_data', return_value=(raw,timestamps,graph)):
                generate(folder,args)
            output = folder/'final_graphs'
            manifest = json.loads((output/'figure_manifest.json').read_text())
            self.assertEqual(len(manifest['figures']),4)
            for name in manifest['figures']:
                self.assertTrue((output/(name+'.png')).is_file())
                self.assertTrue((output/(name+'.pdf')).is_file())
            from pypdf import PdfReader
            self.assertEqual(len(PdfReader(output/'all_graphs.pdf').pages),4)
            profile = pd.read_csv(output/'traffic_24h.csv')
            self.assertEqual(len(profile),288)
            positions = pd.DatetimeIndex(pd.to_datetime(timestamps)).get_indexer(pd.to_datetime(profile['timestamp']))
            np.testing.assert_allclose(profile['actual_mph'],raw[positions,0],rtol=1e-6)
            self.assertTrue((profile['lower95_mph'] < profile['prediction_mph']).all())
            robustness = json.loads((output/'sensor_dropout_results.json').read_text())
            self.assertEqual(len(robustness['rows']),5)


if __name__ == '__main__':
    unittest.main()
