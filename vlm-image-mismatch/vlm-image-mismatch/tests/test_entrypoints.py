import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from reproduce import ROOT, MODELS, build_commands, input_file


def args(**overrides):
    values = dict(task='mismatch', model='qwen3-32b', data_root=ROOT / 'data',
                  output_root=ROOT / 'outputs', profile='full', split='validation',
                  links=ROOT / 'data/retrieval_links.json', mode='gated', max_attach=3)
    values.update(overrides)
    return SimpleNamespace(**values)


class EntryPoints(unittest.TestCase):
    def test_all_tasks_reference_existing_scripts(self):
        for task in ('mismatch', 'direct-query', 'controls', 'retrieval-download',
                     'retrieval-fit', 'retrieval-test'):
            for command in build_commands(args(task=task)):
                self.assertTrue(Path(command[1]).is_file(), command[1])

    def test_model_transfer_arguments(self):
        for key, (model_id, family, label) in MODELS.items():
            command = build_commands(args(model=key))[0]
            for flag, value in (('--model-id', model_id), ('--model-family', family),
                                ('--run-label', label)):
                self.assertEqual(command[command.index(flag) + 1], value)

    def test_paper_prompt_and_quick_separation(self):
        full = build_commands(args(task='direct-query'))[0]
        quick = build_commands(args(task='direct-query', profile='quick'))[0]
        self.assertEqual(full[full.index('--variants') + 1], 'V2')
        self.assertNotEqual(full[full.index('--output-dir') + 1], quick[quick.index('--output-dir') + 1])
        self.assertNotIn('--limit', full)
        self.assertIn('--limit', quick)

    def test_retrieval_modes_are_distinct(self):
        baseline = build_commands(args(task='retrieval-test', mode='baseline'))[0]
        naive = build_commands(args(task='retrieval-test', mode='naive'))[0]
        gated = build_commands(args(task='retrieval-test', mode='gated'))[0]
        self.assertNotIn('--retrieval-manifest', baseline)
        self.assertIn('--attach-always', naive)
        self.assertNotIn('--retrieval-probe', naive)
        self.assertEqual(gated[gated.index('--threshold-policy') + 1], 'youden')
        self.assertIn('--retrieval-probe', gated)
        self.assertNotIn('--attach-always', gated)

    def test_retrieval_requires_full_and_reported_model(self):
        for overrides in ({'profile': 'quick'}, {'model': 'qwen3-8b'}):
            with self.assertRaises(ValueError):
                build_commands(args(task='retrieval-fit', **overrides))

    def test_manifest_is_preferred_over_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(input_file(root, 'train'), root / 'train.json')
            csv = root / 'ablation_manifest_train_portable.csv'
            csv.touch()
            self.assertEqual(input_file(root, 'train'), csv)

    def test_config_has_only_reported_export(self):
        config = json.loads((ROOT / 'mismatch_routing_config.json').read_text())
        self.assertEqual([e['type'] for e in config['experiments']], ['mismatch_activation_export'])
        self.assertTrue(config['experiments'][0]['strict_length_match'])

    def test_random_control_rng_prefix_is_preserved(self):
        command = build_commands(args(task='controls'))[0]
        self.assertEqual(command[command.index('--random-single-heads') + 1], '40')
        self.assertEqual(command[command.index('--random-head-sets') + 1], '20')


if __name__ == '__main__':
    unittest.main()
