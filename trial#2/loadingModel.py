from PuttingEverythingTogether import MODEL_SAVE_PATH
from LinearRegressionModel2 import LinearRegressionModel2
import torch

torch.manual_seed(42)

loaded_model_1 = LinearRegressionModel2()

print(f"Initial dict: {loaded_model_1.state_dict()} ")

loaded_model_1.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

print(f"Loaded state dict: {loaded_model_1.state_dict()}")
