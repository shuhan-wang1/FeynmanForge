from training import PPOTrainer
from feynman_env import FeynmanDiagramEnv
from models import FeynmanGCPN
import torch

env = FeynmanDiagramEnv(['e','e'],['mu','mu'], max_vertices=10, max_steps=50)
model = FeynmanGCPN(9,21,128,3,4,50,10,1.0)
trainer = PPOTrainer(env,model,device='cpu')
trainer._save_current_diagram()
print("Diagram saved!")
