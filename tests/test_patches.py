"""Optional integration tests on pristine files from pinned upstream Git commits."""
import ast
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tap_foley import mmaudio, omni2sound

ROOT = Path(__file__).resolve().parents[1]


def load_function(path, name, namespace):
    node = next(item for item in ast.parse(path.read_text()).body
                if isinstance(item, ast.FunctionDef) and item.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[name]


class PatchTests(unittest.TestCase):
    def check_backend(self, backend, env_name, files, name):
        checkout = os.environ.get(env_name)
        if not checkout:
            self.skipTest(f'Set {env_name} to test the upstream patch')
        spec = json.loads((ROOT / 'patches/upstream.json').read_text())[backend]
        with tempfile.TemporaryDirectory(prefix='tap-patch-test-') as temporary:
            destination = Path(temporary)
            originals = {}
            for relative in files:
                content = subprocess.check_output(['git', '-C', checkout, 'show',
                    f"{spec['commit']}:{relative}"])
                file = destination / relative
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(content)
                originals[relative] = content
            subprocess.run(['git', '-C', str(destination), 'init', '-q'], check=True)
            command = ['git', '-C', str(destination), 'apply']
            patch_file = str(ROOT / 'patches' / spec['patch'])
            subprocess.run(command + ['--check', patch_file], check=True)
            subprocess.run(command + [patch_file], check=True)
            subprocess.run(command + ['--reverse', '--check', patch_file], check=True)
            for relative in files:
                ast.parse((destination / relative).read_text())
            namespace = {'torch': torch}
            modified = load_function(destination / files[0], name, namespace)
            original_file = destination / 'original.py'
            original_file.write_bytes(originals[files[0]])
            original = load_function(original_file, name, {'torch': torch})
            if backend == 'mmaudio':
                marker = object()
                kwargs = dict(feature_utils=None, net=None, fm=None, rng=None, cfg_strength=4.5)
                with patch.object(mmaudio, 'generate', return_value=marker) as mocked:
                    result = modified(None, None, ['target'], source_text=['source'],
                                      sampling_method='tap-foley', **kwargs)
                    self.assertIs(result, marker)
                    self.assertEqual(mocked.call_args.kwargs['source_text'], ['source'])
                    self.assertEqual(mocked.call_args.kwargs['alpha'], 1.2)
                # Execute native default branch with tiny fake backbone/features.
                features = SimpleNamespace(device='cpu', dtype=torch.float32,
                    encode_text=lambda texts: torch.ones(len(texts), 2, 3),
                    decode=lambda x: x, vocode=lambda x: x)
                net = SimpleNamespace(latent_seq_len=2, latent_dim=3,
                    get_empty_clip_sequence=lambda bs: torch.zeros(bs, 2, 3),
                    get_empty_sync_sequence=lambda bs: torch.zeros(bs, 2, 3),
                    get_empty_string_sequence=lambda bs: torch.zeros(bs, 2, 3),
                    preprocess_conditions=lambda *args: args,
                    get_empty_conditions=lambda *args, **kw: None,
                    ode_wrapper=lambda t, x, *args: x*.1, unnormalize=lambda x: x)
                fm = SimpleNamespace(to_data=lambda fn, x: x + fn(0, x))
                def run(fn):
                    return fn(None, None, ['target'], feature_utils=features, net=net, fm=fm,
                              rng=torch.Generator().manual_seed(42), cfg_strength=4.5)
                torch.testing.assert_close(run(modified), run(original), rtol=0, atol=0)
            else:
                marker = object()
                with patch.object(omni2sound, 'generate', return_value=marker) as mocked:
                    result = modified(None, sampling_method='tap-foley', source_prompt='source',
                                      steps=100, cfg_scale=2.5)
                    self.assertIs(result, marker)
                    self.assertEqual(mocked.call_args.kwargs['alpha'], 1.2)
                    self.assertEqual(mocked.call_args.kwargs['steps'], 100)
                with self.assertRaises(ValueError):
                    modified(None, sampling_method='tap-foley', sampler_type='dpmpp-2m-sde')
                def sample_k(model, noise, init_audio, steps, **kwargs):
                    return noise + kwargs['cross_attn_cond'].mean()
                for fn in (modified, original):
                    fn.__globals__['sample_k'] = sample_k
                model = SimpleNamespace(pretransform=None, io_channels=2,
                    model=torch.nn.Linear(2, 2), diffusion_objective='v',
                    get_conditioning_inputs=lambda cond: {'cross_attn_cond': cond['text'][0]})
                kwargs = dict(conditioning_tensors={'text': [torch.ones(1, 2, 3), None]},
                              sample_size=16, seed=42, device='cpu', return_latents=True)
                torch.testing.assert_close(modified(model, **kwargs), original(model, **kwargs),
                                           rtol=0, atol=0)
            subprocess.run(command + ['--reverse', patch_file], check=True)
            for relative in files:
                self.assertEqual((destination / relative).read_bytes(), originals[relative])

    def test_mmaudio_patch(self):
        self.check_backend('mmaudio', 'TAP_MMAUDIO_ROOT', ['mmaudio/eval_utils.py', 'demo.py'], 'generate')

    def test_omni2sound_patch(self):
        self.check_backend('omni2sound', 'TAP_OMNI2SOUND_ROOT', ['pyscripts/generate_v2a_cond.py'], 'generate_diffusion_cond')


if __name__ == '__main__':
    unittest.main()
