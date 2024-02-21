import os
import unittest
import torch
from helpers import tests_path
from voicebox import Voicebox
from voicebox_pipeline import VoiceboxPipeline

torch.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TestInference(unittest.TestCase):
    
    def test_from_pretrained(self):
        VoiceboxPipeline.from_pretrained(model_path = os.path.join(tests_path(), "voicebox.pt"), device = device)