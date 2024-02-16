import unittest
import torch
from voicebox_pipeline import VoiceboxPipeline

torch.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TestMatchTTS(unittest.TestCase):
    
    def test_inference(self):
        pipe = VoiceboxPipeline()
        pipe()