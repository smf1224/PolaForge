from thop import profile
import torch
from main import get_args_parser
from models import build_model
import argparse
from util.misc import NestedTensor
from models.decoder import Decoder, build
parser = argparse.ArgumentParser('PolaForge', parents=[get_args_parser()])
args = parser.parse_args()

if __name__ == '__main__':
    model, criterion = build(args)
    model.to(args.device)

    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters")
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Total Parameters: {total_params}")
    input1 = torch.randn(1, 3, 512, 512).to(torch.device(args.device))
    input2 = torch.randn(1, 3, 512, 512).to(torch.device(args.device))
    input3 = torch.randn(1, 3, 512, 512).to(torch.device(args.device))
    input4 = torch.randn(1, 3, 512, 512).to(torch.device(args.device))
    input5 = torch.randn(1, 3, 512, 512).to(torch.device(args.device))
    input6 = torch.randn(1, 3, 512, 512).to(torch.device(args.device))
    input_list = [input1, input2, input3, input4, input5, input6]
    modal_imgs = [NestedTensor(img, torch.ones(img.shape[0], img.shape[2], img.shape[3], dtype=torch.bool).to(args.device)) for img in input_list]
    for i, backbone in enumerate(model.backbones):
        print(f"Backbone {i} type: {type(backbone)}")
    flops, params = profile(model, (modal_imgs,))
    print("flops(G):", flops/1e9, "params(M):", params/1e6)
