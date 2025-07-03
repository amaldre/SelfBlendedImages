from model import Detector
import torch

WEIGHTS = "weights/FFc23.tar"
model=Detector()
cnn_sd=torch.load(WEIGHTS)["model"]
model.load_state_dict(cnn_sd)
model = model.cuda()

for name, module in model.named_modules():
    print(name)
