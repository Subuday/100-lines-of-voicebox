import unittest
import torch
from voicebox import Voicebox
from flow_matching_pipeline import FlowMachingPipeline

torch.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TestFlowMachingPipeline(unittest.TestCase):

    @staticmethod
    def _create_inputs(batch_size):
        y = torch.randint(1, 1000, (batch_size, 767)).to(device)
        x = torch.rand(batch_size, 767, 128).to(device)
        return y, x
    

    def _test_forward(self, batch_size):
        y, x = self._create_inputs(batch_size)
        pipeline = FlowMachingPipeline().to(device)
        pipeline.train()
        
        pipeline.forward(y, x)


    def test_forward(self):
        self._test_forward(1)
        self._test_forward(3)

    # TODO: Remove this test
    def test_inference(self):
        model = Voicebox()
        model.to(device)
        pipe = FlowMachingPipeline(model = model)
        pipe.inference()