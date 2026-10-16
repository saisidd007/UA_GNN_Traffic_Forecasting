import unittest
from unittest.mock import patch

import numpy as np

from generate_results_pdf import persistence_rmse_by_horizon


class PersistenceHorizonTests(unittest.TestCase):
    def test_last_input_alignment_and_each_horizon(self):
        values=np.broadcast_to(np.arange(50,dtype=np.float32)[:,None],(50,2)).copy()
        observed=np.ones_like(values,dtype=bool)
        scale=np.ones(2)
        indices=np.array([0,1])
        prepared=(values,observed,np.zeros(2),scale,{'test':indices},(20,30))
        config=dict(dataset='METR-LA',train_ratio=.7,val_ratio=.1,test_windows=2)
        baseline=dict(observed_test_targets=48,rmse_mph=float(np.sqrt(np.mean(np.arange(1,13)**2))),
                      mae_by_horizon_mph=list(range(1,13)))
        with patch('generate_results_pdf.load_data',return_value=(values,[],None)), \
             patch('generate_results_pdf.prepare',return_value=prepared):
            actual=persistence_rmse_by_horizon(config,baseline)
            np.testing.assert_allclose(actual,np.arange(1,13))
            baseline['observed_test_targets']=47
            with self.assertRaisesRegex(ValueError,'differs'):
                persistence_rmse_by_horizon(config,baseline)


if __name__=='__main__':
    unittest.main()
