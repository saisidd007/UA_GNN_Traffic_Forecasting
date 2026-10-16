from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from generate_seed_reports import main
from report_run_paths import dataset_run


class SeedReportTests(unittest.TestCase):
    def test_existing_layout_and_nested_reference_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            self.assertEqual(dataset_run(root,'METR-LA',42),root/'METR-LA')
            for seed in [123,2026]:
                folder=root/'METR-LA'/f'seed_{seed}'/'reference'
                folder.mkdir(parents=True)
                (folder/'metrics.json').write_text('{}')
                self.assertEqual(dataset_run(root,'METR-LA',seed),folder)

    def test_wrapper_requests_both_reports_without_training(self):
        with patch('sys.argv',['generate_seed_reports.py']), \
             patch('generate_seed_reports.subprocess.run') as run, redirect_stdout(io.StringIO()):
            main()
        self.assertEqual(run.call_count,2)
        for call,seed in zip(run.call_args_list,[123,2026]):
            command=call.args[0]
            self.assertTrue(command[2].endswith('generate_results_pdf.py'))
            self.assertEqual(command[command.index('--seed')+1],str(seed))
            self.assertIn('--generate-graphs',command)
            self.assertIn('METR-LA',command)
            self.assertIn('PEMS-BAY',command)


if __name__=='__main__':
    unittest.main()
