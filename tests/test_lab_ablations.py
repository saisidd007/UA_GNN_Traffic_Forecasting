import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
import io
from unittest.mock import patch

from run_lab_ablations import completed, expected, make_plan, write_summary


class LabAblationTests(unittest.TestCase):
    def test_single_default_command_includes_sensor_stage(self):
        from run_lab_ablations import main
        with tempfile.TemporaryDirectory() as temporary:
            with patch('sys.argv',['run_lab_ablations.py','--output-root',temporary]), \
                 patch('run_lab_ablations.completed',return_value=True), \
                 patch('run_lab_ablations.write_summary'), \
                 patch('torch.cuda.is_available',return_value=True), \
                 patch('run_lab_ablations.subprocess.run') as launched, redirect_stdout(io.StringIO()):
                main()
            launched.assert_called_once()
            command=launched.call_args.args[0]
            self.assertTrue(command[2].endswith('run_lab_sensor_failures.py'))
            self.assertIn('METR-LA',command)
            self.assertIn('PEMS-BAY',command)
            self.assertIn(str(Path(temporary)/'sensor_failures'),command)
            self.assertEqual(json.loads((Path(temporary)/'runner_status.json').read_text())['state'],'completed')

    def test_plan_reuses_existing_seed42_and_adds_two_new_seeds(self):
        tasks=make_plan(Path('/new'),Path('/existing'),['METR-LA','PEMS-BAY'])
        self.assertEqual(len(tasks),24)
        reused=[t for t in tasks if t['reuse_existing']]
        self.assertEqual(len(reused),2)
        self.assertTrue(all(t['seed']==42 and t['variant']=='reference' for t in reused))
        self.assertEqual({t['seed'] for t in tasks if t['seed']!=42},{123,2026})
        self.assertEqual(len({t['folder'] for t in tasks}),24)

    def test_resume_rejects_unmatched_runs_and_summary_sd(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            tasks=make_plan(root/'new',root/'existing',['METR-LA'])
            for task in tasks:
                folder=Path(task['folder'])
                folder.mkdir(parents=True)
                metrics=expected(task['dataset'],task['seed'],task['variant'])
                metrics.update(epochs_completed=50,mae_mph={42:1.,123:2.,2026:3.}[task['seed']],
                               rmse_mph=4.,coverage_95=.95,width_95_mph=10.,best_epoch=45)
                (folder/'metrics.json').write_text(json.dumps(metrics))
                (folder/'completed.json').write_text(json.dumps(dict(epochs_completed=50)))
                for name in ['predictions.npz','best.pt','training.json']:
                    (folder/name).touch()
            summary=write_summary(root,tasks)
            self.assertEqual(summary['completed_runs'],12)
            self.assertTrue(all(row['mae_mph_mean']==2 and row['mae_mph_sd']==1 for row in summary['aggregate']))
            folder=Path(tasks[0]['folder'])
            metrics=json.loads((folder/'metrics.json').read_text())
            metrics['batch_size']=64
            (folder/'metrics.json').write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError,'batch_size'):
                completed(folder,expected(tasks[0]['dataset'],tasks[0]['seed'],tasks[0]['variant']))
            (folder/'completed.json').unlink()
            self.assertFalse(completed(folder,expected(tasks[0]['dataset'],tasks[0]['seed'],tasks[0]['variant'])))


if __name__=='__main__':
    unittest.main()
