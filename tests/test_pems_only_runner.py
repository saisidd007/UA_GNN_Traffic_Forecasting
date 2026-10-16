from pathlib import Path
import unittest

from run_pems_ablations_42_123 import make_tasks


class PemsOnlyRunnerTests(unittest.TestCase):
    def test_scope_and_original_checkpoint_paths(self):
        tasks=make_tasks(Path('/runs'),Path('/old'))
        self.assertEqual(len(tasks),6)
        self.assertEqual({t['seed'] for t in tasks},{42,123})
        self.assertEqual({t['variant'] for t in tasks},{'no_diffusion','no_dropout','no_aleatoric'})
        for task in tasks:
            self.assertEqual(task['folder'],Path('/runs')/'PEMS-BAY'/f"seed_{task['seed']}"/task['variant'])
            self.assertNotIn('METR-LA',str(task['folder']))
            self.assertEqual(task['reference'],Path('/old/PEMS-BAY') if task['seed']==42 else Path('/runs/PEMS-BAY/seed_123/reference'))


if __name__=='__main__':
    unittest.main()
