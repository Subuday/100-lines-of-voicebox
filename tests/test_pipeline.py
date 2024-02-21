import unittest
import torch
from voicebox import Voicebox
from flow_matching_pipeline import FlowMachingPipeline

torch.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TestMatchTTS(unittest.TestCase):
    
    def test_inference(self):
        model = Voicebox()
        model.to(device)
        pipe = FlowMachingPipeline(model = model)
        pipe.inference()