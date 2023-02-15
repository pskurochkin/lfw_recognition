import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50


class LFWEmbedder(nn.Module):
    def __init__(self, backbone, input_dims, out_dims):
        super(LFWEmbedder, self).__init__()

        self.backbone = backbone.eval()

        self.fc1 = nn.Linear(input_dims, 256, bias=False)
        self.fc1_norm = nn.BatchNorm1d(256)
        self.fc1_relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(256, 256, bias=False)
        self.fc2_norm = nn.BatchNorm1d(256)
        self.fc2_relu = nn.ReLU()

        self.fc_out = nn.Linear(256, out_dims)

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)

        x = self.fc1(x)
        x = self.fc1_norm(x)
        x = self.fc1_relu(x)

        x = self.dropout(x)

        x = self.fc2(x)
        x = self.fc2_norm(x)
        x = self.fc2_relu(x)

        output = self.fc_out(x)

        return F.normalize(output, p=2)

    def train(self, mode=True):
        self.training = mode

        self.backbone.eval()
        for module in self.children():
            if module is not self.backbone:
                module.train(mode)
        return self

def create_model(embedding_dims=256):
    model = resnet50(pretrained=True)
    model.fc = nn.Identity()

    return LFWEmbedder(model, input_dims=2048, out_dims=embedding_dims)
