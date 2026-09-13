import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from scipy.io import wavfile

try:
    from src.language import transcribe as tr
except ModuleNotFoundError:
    import transcribe as tr


class TranscribeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def audio(self, name="call.wav", rate=8000):
        time = np.arange(rate) / rate
        caller = 0.5*np.sin(2*np.pi*440*time)
        agent = 0.8*np.sin(2*np.pi*1200*time)
        path = self.root/name
        wavfile.write(path, rate, (np.column_stack([caller, agent])*32767).astype(np.int16))
        return path

    def test_real_resampling_preserves_duration_pitch_and_caller(self):
        for rate in [8000, 16000, 44100]:
            with self.subTest(rate=rate):
                audio, original, duration = tr.prepare_audio(self.audio(rate=rate))
                self.assertEqual(original, rate)
                self.assertEqual(duration, 1)
                self.assertEqual(audio.shape, (16000,))
                self.assertEqual(audio.dtype, np.float32)
                fft = abs(np.fft.rfft(audio))
                self.assertEqual(np.argmax(fft), 440)
                self.assertLess(fft[1200], fft[440]*0.01)
                self.assertAlmostEqual(float(np.max(audio)), 0.5, places=2)

    def test_float_mono_is_not_divided_by_32768(self):
        path = self.root/'float.wav'
        wavfile.write(path, 16000, np.full(160, 0.25, dtype=np.float32))
        audio, _, _ = tr.prepare_audio(path)
        np.testing.assert_allclose(audio, 0.25)

    def test_unsigned_pcm_centered(self):
        path = self.root/'unsigned.wav'
        wavfile.write(path, 16000, np.full(160, 128, dtype=np.uint8))
        audio, _, _ = tr.prepare_audio(path)
        np.testing.assert_array_equal(audio, 0)

    def test_three_statuses_and_lazy_errors(self):
        path = self.audio()
        model = Mock()
        model.transcribe.return_value = (iter([SimpleNamespace(text=" Hola mundo. ")]), None)
        result = tr.process_single_audio(path, model, 'test')
        self.assertEqual((result['status'],result['word_count']), ('success',2))
        model.transcribe.return_value = (iter([]), None)
        self.assertEqual(tr.process_single_audio(path,model,'test')['status'],'empty')
        def broken():
            yield SimpleNamespace(text='partial')
            raise RuntimeError('decoder failed')
        model.transcribe.return_value = (broken(), None)
        result = tr.process_single_audio(path,model,'test')
        self.assertEqual(result['status'],'error')
        self.assertIn('decoder failed',result['error'])
        self.assertEqual(result['caller_transcript'],'')

    def test_model_loaded_once_checkpoints_resume_and_retry(self):
        files = [self.audio('a.wav'), self.root/'missing.wav', self.audio('b.wav')]
        model = Mock()
        model.transcribe.side_effect = lambda *a,**k: (iter([SimpleNamespace(text='Hola mundo')]),None)
        factory=Mock(return_value=model)
        output=self.root/'corrected.csv'
        previous=[{'file_name':'a.wav','caller_transcript':''}, {'file_name':'b.wav','caller_transcript':'Antes'}]
        report=tr.run_batch(files,output,previous,'test',model_factory=factory)
        factory.assert_called_once()
        self.assertEqual(model.transcribe.call_count,2)
        self.assertEqual((report['new_success'],report['new_errors']), (2,1))
        self.assertEqual(report['old_empty'],1)
        original=output.read_bytes()
        no_load=Mock(side_effect=AssertionError('Must not load'))
        tr.run_batch(files,output,previous,'test',model_factory=no_load)
        self.assertEqual(output.read_bytes(),original)
        no_load.assert_not_called()
        self.audio('missing.wav')
        retry=Mock(return_value=model)
        tr.run_batch(files,output,previous,'test',retry_errors=True,model_factory=retry)
        retry.assert_called_once()
        self.assertEqual(model.transcribe.call_count,3)
        self.assertEqual(len(tr.read_csv(output)),3)

    def test_reject_incompatible_checkpoint(self):
        output=self.root/'old.csv'
        output.write_text('file_name,caller_transcript\na.wav,old\n')
        with self.assertRaisesRegex(ValueError,'compatible'):
            tr.run_batch([],output,[])

    def test_interrupted_batch_preserves_completed_calls(self):
        files = [self.audio('a.wav'), self.audio('b.wav')]
        model = Mock()
        model.transcribe.side_effect = [(iter([SimpleNamespace(text='Hola')]), None), KeyboardInterrupt()]
        output = self.root/'interrupted.csv'
        with self.assertRaises(KeyboardInterrupt):
            tr.run_batch(files, output, [], 'test', model_factory=Mock(return_value=model))
        self.assertEqual([r['file_name'] for r in tr.read_csv(output)], ['a.wav'])
        model.transcribe.side_effect = lambda *a, **k: (iter([]), None)
        tr.run_batch(files, output, [], 'test', model_factory=Mock(return_value=model))
        self.assertEqual(model.transcribe.call_count, 3)
        self.assertEqual([r['status'] for r in tr.read_csv(output)], ['success', 'empty'])

    def test_original_output_is_protected(self):
        from unittest.mock import patch
        import sys
        original = self.root/'src/language/caller_transcriptions.csv'
        original.parent.mkdir(parents=True)
        original.write_text('file_name,caller_transcript\na.wav,original\n')
        before = original.read_bytes()
        with patch.object(sys, 'argv', ['transcribe', '--root', str(self.root), '--output', str(original)]):
            with self.assertRaises(SystemExit) as error:
                tr.main()
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(original.read_bytes(), before)

    def test_empty_and_nonfinite_audio(self):
        path=self.root/'bad.wav'
        for samples in [np.array([],dtype=np.float32),np.array([np.nan],dtype=np.float32)]:
            wavfile.write(path,8000,samples)
            with self.assertRaises(ValueError):tr.prepare_audio(path)


if __name__ == '__main__':
    unittest.main()
