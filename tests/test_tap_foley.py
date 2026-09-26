import math
import unittest
from types import SimpleNamespace

import torch
from tap_foley import project
from tap_foley import mmaudio, omni2sound


class ProjectionTests(unittest.TestCase):
    def test_active_inactive_degenerate_and_batch_independence(self):
        base = torch.tensor([[[2., 3.]], [[-1., 4.]], [[8., 9.]]])
        source = torch.tensor([[[1., 0.]], [[1., 0.]], [[0., 0.]]])
        target = torch.zeros_like(base)
        result = project(base, source, target)
        torch.testing.assert_close(result, torch.tensor([[[-.4, 3.]], [[-1., 4.]], [[8., 9.]]]))
        for index in range(3):
            torch.testing.assert_close(result[index:index+1], project(
                base[index:index+1], source[index:index+1], target[index:index+1]))

    def test_random_halfspace_and_orthogonal_residual(self):
        torch.manual_seed(7)
        base, source, target = (torch.randn(5, 7, 9) for _ in range(3))
        out = project(base, source, target, alpha=1.2)
        direction = source - target
        boundary = ((out - target) * direction).flatten(1).sum(1)
        self.assertTrue((boundary <= 1e-4).all())
        correction = base - out
        perpendicular = correction - ((correction * direction).flatten(1).sum(1)
            / direction.square().flatten(1).sum(1))[:, None, None] * direction
        self.assertLess(perpendicular.abs().max().item(), 2e-6)
        # alpha=1 is the exact nearest feasible point for active samples.
        nearest = project(base, source, target, alpha=1)
        torch.testing.assert_close(project(nearest, source, target, alpha=1), nearest)

    def test_dtype_and_validation(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            x = torch.ones(2, 3, 4, dtype=dtype)
            self.assertEqual(project(x, x, x).dtype, dtype)
        for alpha in (0, .9, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                project(x, x, x, alpha=alpha)
        with self.assertRaises(ValueError):
            project(x * float('nan'), x, x)
        with self.assertRaises(ValueError):
            project(x, x[:1], x)


class FakeMMAudio:
    def predict_flow(self, x, t, condition):
        return torch.zeros_like(x) + condition


class SamplerTests(unittest.TestCase):
    def test_mmaudio_euler_constant_field(self):
        noise = torch.tensor([[[.2, -.3]]])
        branches = dict(empty=0., video_target=2., source=1., target=0.)
        result = mmaudio.sample(FakeMMAudio(), noise, branches, steps=25, cfg=4.5)
        torch.testing.assert_close(result, noise - 1.8)

    def test_omni_rotation_and_cfg_bypass(self):
        calls = []
        def model(x, t, value, **kwargs):
            calls.append((t.clone(), kwargs))
            return torch.zeros_like(x) + value
        branches = {k: {'value': v} for k, v in
                    dict(empty=0., video_target=2., source=1., target=0.).items()}
        noise = torch.tensor([[[.2, -.3]]])
        result = omni2sound.sample(model, noise, branches, steps=100, cfg=2.5)
        # Constant projected v=-1; closed form of x <- cos(h)x - sin(h)v.
        c, s = math.cos(math.pi/200), math.sin(math.pi/200)
        expected = c**100 * noise + s * (1-c**100)/(1-c)
        torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-5)
        self.assertEqual(len(calls), 400)
        self.assertTrue(all(kwargs['cfg_scale'] == 1 and kwargs['scale_phi'] == 0
                            for _, kwargs in calls))
        for index in range(0, len(calls), 4):
            self.assertTrue(all(torch.equal(calls[index][0], calls[j][0])
                                for j in range(index, index + 4)))

    def test_invalid_steps(self):
        for steps in (0, -1, 2.5):
            with self.assertRaises(ValueError):
                mmaudio.sample(None, torch.ones(1, 2, 3), {}, steps=steps)
            with self.assertRaises(ValueError):
                omni2sound.sample(None, torch.ones(1, 2, 3), {}, steps=steps)

    def test_omni_conditioning_contract(self):
        class Model:
            def __init__(self):
                self.calls = []
            def get_conditioning_inputs(self, values):
                self.calls.append(values)
                return {'cross_attn_cond': torch.ones(1, 4, 2),
                        'additional_cond': values['sync_feature'][0],
                        'global_cond': values['seconds_total'][0]}
        model = Model()
        pair = lambda value: [torch.full((1, 2, 3), value), torch.ones(1, 2, dtype=torch.bool)]
        source = {k: pair(i+1.) for i, k in enumerate(
            ['clip_feature', 'sync_feature', 'text_prompt', 'seconds_start', 'seconds_total'])}
        target = pair(10.)
        branches = omni2sound.build_branches(model, source, target)
        self.assertTrue((branches['empty']['cross_attn_cond'] == 0).all())
        self.assertTrue((branches['empty']['additional_cond'] == 0).all())
        torch.testing.assert_close(branches['empty']['global_cond'], source['seconds_total'][0])
        for values in model.calls[1:3]:
            self.assertNotIn('clip_feature', values)
            self.assertTrue((values['sync_feature'][0] == 0).all())
            self.assertTrue(values['sync_feature'][1].all())
        self.assertIs(model.calls[3]['clip_feature'], source['clip_feature'])
        self.assertIs(model.calls[3]['text_prompt'], target)
        self.assertTrue((source['sync_feature'][0] != 0).all())

    def test_mmaudio_generation_uses_learned_nulls(self):
        class Net(FakeMMAudio):
            latent_seq_len, latent_dim = 2, 3
            def __init__(self): self.conditions = []
            def get_empty_clip_sequence(self, bs): return torch.full((bs, 2, 3), -7.)
            def get_empty_sync_sequence(self, bs): return torch.full((bs, 2, 3), -8.)
            def get_empty_conditions(self, bs): return 0.
            def preprocess_conditions(self, clip, sync, text):
                self.conditions.append((clip, sync, text))
                return float(text[0, 0, 0])
            def unnormalize(self, x): return x
        class Features:
            device, dtype = 'cpu', torch.float32
            def encode_video_with_clip(self, x, **kwargs): return x
            def encode_video_with_sync(self, x, **kwargs): return x
            def encode_text(self, texts): return torch.full((len(texts), 2, 3), 1. if texts[0]=='source' else 2.)
            def decode(self, x): return x
            def vocode(self, x): return x
        net, features = Net(), Features()
        result = mmaudio.generate(torch.ones(1, 2, 3), torch.ones(1, 2, 3), ['target'],
            source_text=['source'], feature_utils=features, net=net,
            fm=SimpleNamespace(inference_mode='euler', min_sigma=0, num_steps=25),
            rng=torch.Generator().manual_seed(42))
        self.assertEqual(result.shape, (1, 2, 3))
        for clip, sync, _ in net.conditions[1:]:
            self.assertTrue((clip == -7).all())
            self.assertTrue((sync == -8).all())

    def test_omni_generation_seed_batch_and_decode(self):
        class Diffusion(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))
            def forward(self, x, t, cross_attn_cond, **kwargs):
                return x * .05 + cross_attn_cond.mean(dim=(1, 2))[:, None, None]
        class Decoder(torch.nn.Module):
            downsampling_ratio = 2
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))
            def decode(self, x):
                return x.repeat_interleave(2, dim=-1)
        model = SimpleNamespace(model=Diffusion(), pretransform=Decoder(),
            diffusion_objective='v', io_channels=2,
            get_conditioning_inputs=lambda values: {
                'cross_attn_cond': values['text_prompt'][0],
                'additional_cond': values['sync_feature'][0],
                'global_cond': values['seconds_total'][0]})
        pair = lambda value: [torch.full((2, 2, 3), value), torch.ones(2, 2, dtype=torch.bool)]
        conditions = {key: pair(i+1.) for i, key in enumerate(
            ['clip_feature', 'sync_feature', 'text_prompt', 'seconds_start', 'seconds_total'])}
        kwargs = dict(conditioning_tensors=conditions, source_text_condition=pair(-1.),
                      sample_size=16, seed=42, device='cpu', steps=3)
        latent = omni2sound.generate(model, **kwargs, return_latents=True)
        audio = omni2sound.generate(model, **kwargs)
        self.assertEqual(latent.shape, (2, 2, 8))
        self.assertEqual(audio.shape, (2, 2, 16))
        torch.testing.assert_close(audio, latent.repeat_interleave(2, dim=-1), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            omni2sound.generate(model, **{**kwargs, 'sample_size': 15})


if __name__ == '__main__':
    unittest.main()
