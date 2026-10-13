import argparse
from contextlib import redirect_stdout
import io
import unittest

import numpy as np
import torch

from run_lab_sensor_failures import evaluate_case, failure_mask


class LastReadingModel(torch.nn.Module):
    def forward(self,x,mask,tod,dow,forward,reverse,edges):
        mean=x[:,-1,:,None].expand(-1,-1,12)
        return mean,torch.ones_like(mean)


class SensorFailureTests(unittest.TestCase):
    def test_persistent_and_random_masks_and_nested_failure_rates(self):
        persistent=np.linspace(0,1,100)
        mask=failure_mask(np.random.default_rng(5),'persistent',8,100,.3,persistent)
        self.assertTrue(np.all(mask==mask[0]))
        first=failure_mask(np.random.default_rng(5),'random',8,100,.1,persistent)
        second=failure_mask(np.random.default_rng(5),'random',8,100,.5,persistent)
        self.assertTrue(np.all(first<=second))
        self.assertFalse(np.all(second==second[0]))

    def test_failure_evaluation_keeps_targets_and_computes_rmse(self):
        values=np.full((100,3),10,dtype=np.float32)
        observed=np.ones_like(values,dtype=bool)
        indices=np.arange(8)
        clock=np.zeros(100,dtype=np.int64)
        args=argparse.Namespace(inference_seed=42,batch_size=2,mc_passes=2)
        with redirect_stdout(io.StringIO()):
            clean=evaluate_case(LastReadingModel(),torch.device('cpu'),values,observed,np.ones(3)*2,
                                indices,clock,clock,None,None,None,'random',0,17,args)
            failed=evaluate_case(LastReadingModel(),torch.device('cpu'),values,observed,np.ones(3)*2,
                                 indices,clock,clock,None,None,None,'persistent',1,17,args)
        self.assertEqual(clean['mae_mph'],0)
        self.assertEqual(clean['rmse_mph'],0)
        self.assertEqual(failed['mae_mph'],20)
        self.assertEqual(failed['rmse_mph'],20)
        self.assertEqual(clean['observed_test_targets'],failed['observed_test_targets'])
        self.assertEqual(failed['observed_test_targets'],8*3*12)


if __name__=='__main__':
    unittest.main()
