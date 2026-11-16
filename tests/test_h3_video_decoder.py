"""Decoder selection must not change the encoder, audio or inference recipe."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('decoder_test', ROOT / 'iamccs_h3_video_decoder.py')
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


class DecoderTests(unittest.TestCase):
    def test_original_is_identity_and_does_not_load_models(self):
        original = object()
        p = {'sampling': {'seed': 9, 'steps': 6}, 'audio': {'mode': 'locked'}}
        D.configure(p, {}, [])
        loader = Mock(side_effect=AssertionError('Native must not load a second VAE'))
        with patch.dict(sys.modules, {'nodes': SimpleNamespace(VAELoader=loader)}):
            self.assertIs(D.select_video_decoder(original, p), original)
        self.assertEqual(p['sampling'], {'seed': 9, 'steps': 6})
        self.assertEqual(p['audio'], {'mode': 'locked'})

    def test_sol_automatically_prefers_registered_int8_and_preserves_original(self):
        p = {}
        name = 'nested/minimax_h3_sol_vae_int8_convrot.safetensors'
        D.configure(p, {'h3_video_decoder': 'sol'}, ['other.safetensors', 'minimax_h3_sol_vae_fp16.safetensors', name])
        decoder = SimpleNamespace(latent_channels=24, first_stage_model=SimpleNamespace(noncausal=True))
        loader = Mock(return_value=(decoder,))
        original = SimpleNamespace(encoder=object(), audio=object())
        old_encoder, old_audio = original.encoder, original.audio
        with patch.dict(sys.modules, {'nodes': SimpleNamespace(VAELoader=lambda: SimpleNamespace(load_vae=loader))}):
            self.assertIs(D.select_video_decoder(original, p), decoder)
        loader.assert_called_once_with(vae_name=name)
        self.assertIs(original.encoder, old_encoder)
        self.assertIs(original.audio, old_audio)
        self.assertEqual(p['video_decoder']['scope'], 'native_stage_decode_only')

    def test_missing_and_ambiguous_models_stop_during_planning(self):
        for names in ([], ['sol_h3_a.safetensors', 'sol_h3_b.safetensors']):
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, 'install one'):
                D.configure({}, {'h3_video_decoder': 'sol'}, names)

    def test_invalid_setting_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            D.configure({}, {'h3_video_decoder': 'wrong'}, [])

    def test_wrong_checkpoint_is_rejected(self):
        p = {'video_decoder': {'mode': 'sol', 'model_name': 'sol_h3_wrong.safetensors'}}
        for channels, causal in ((16, False), (24, False)):
            vae = SimpleNamespace(latent_channels=channels, first_stage_model=SimpleNamespace(noncausal=causal))
            with patch.dict(sys.modules, {'nodes': SimpleNamespace(VAELoader=lambda: SimpleNamespace(load_vae=lambda **kw: (vae,)))}):
                with self.assertRaisesRegex(ValueError, 'noncausal 24-channel'):
                    D.select_video_decoder(object(), p)


if __name__ == '__main__':
    unittest.main()
