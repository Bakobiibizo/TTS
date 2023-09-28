import unittest

from TTS.tts.utils.text import phonemes

class SymbolsTest(unittest.TestCase):
    def test_uniqueness(self):  #pylint: disable=no-self-use
        assert sorted(phonemes) == sorted(
            list(set(phonemes))
        ), f" {len(phonemes)} vs {len(set(phonemes))} "
