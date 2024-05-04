import os
import importlib
from erfnet_quantized import Net
import torch
from torch import nn
from torch.ao.quantization import quantize_fx, observer, get_default_qconfig_mapping
# import torch.ao.quantization as quantization
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary
from torchprofile import profile_macs
import torch.profiler as profiler
from ptflops import get_model_complexity_info
from flopth import flopth
from torchstat import stat
from dataset import cityscapes
from main import MyCoTransform
from copy import deepcopy
from argparse import ArgumentParser

backend = "x86"

def calibrate(model, data_load):
    co_transform_val = MyCoTransform(False, augment=False, height=512)
    dataset_val = cityscapes(args.datadir, co_transform_val, 'val')
    loader_val = DataLoader(dataset_val, num_workers=4, batch_size=args.batch_size, shuffle=False)
    print(len(loader_val))
    with torch.inference_mode():
        for step, (images, _) in enumerate(loader_val ):
            print(f"step {step + 1}")
            print(images.shape)
            if not args.cpu:
                images = images.cuda()
            images = Variable(images)
            model(images)

def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            if name.startswith('module.'):
                own_state[name.split('module.')[-1]].copy_(param)
            else:
                print(name, ' not loaded')
                continue
        else:
            own_state[name].copy_(param)
    return model

def load_quant_dict(model, state_dict):
    own_state = model.state_dict()
    for name, params in state_dict.items():
        own_state[name]=deepcopy(params)
    return model

def quantize_model2(model, args, calibrate):
    m = deepcopy(model)
    modules_to_fuse = [[f"encoder.layers.{i}.conv1x3_1", f"encoder.layers.{i}.bn1"] for i in range(1, 6)]
    modules_to_fuse += [[f"encoder.layers.{i}.conv1x3_1", f"encoder.layers.{i}.bn1"] for i in range(7, 15)]
    modules_to_fuse += [[f"decoder.layers.{i}.conv1x3_1", f"decoder.layers.{i}.bn1"] for i in range(1, 3)]
    modules_to_fuse += [[f"decoder.layers.{i}.conv1x3_1", f"decoder.layers.{i}.bn1"] for i in range(4, 6)]
    # for f in modules_to_fuse:
    #     print(f)
    m.eval()
    config = torch.ao.quantization.qconfig.QConfig(
        activation = torch.ao.quantization.MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric),
        weight = torch.ao.quantization.MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.qint8)
    )
    # m.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    m.qconfig = config
    print()
    print(config)
    print()
    fused_model = torch.ao.quantization.fuse_modules(m, modules_to_fuse)
    model_prepared = torch.ao.quantization.prepare(fused_model)
    if calibrate:
        co_transform_val = MyCoTransform(False, augment=False, height=512)
        dataset_val = cityscapes(args.datadir, co_transform_val, 'val')
        loader_val = DataLoader(dataset_val, num_workers=4, batch_size=args.batch_size, shuffle=False)
        print(len(loader_val))
        with torch.inference_mode():
            for step, (images, _) in enumerate(loader_val):
                print(f"step {step + 1}")
                print(images.shape)
                if not args.cpu:
                    images = images.cuda()
                images = Variable(images)
                model_prepared(images)
    model_quantized = torch.ao.quantization.convert(model_prepared, remove_qconfig=False)
    return model_quantized

def quantize_model(model, args, calibrate): #datadir should be the path to the cityscapes validation dataset
    m = deepcopy(model)
    example_input = torch.randn(1, 20, 512, 1024)
    my_qconfig = torch.ao.quantization.QConfig(
        weight = observer.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine),
        activation = observer.MinMaxObserver.with_args(dtype = torch.qint8, qscheme=torch.per_tensor_affine)
    )
    qconfig = get_default_qconfig_mapping(backend)
    if not args.cpu:
        example_input = example_input.cuda()
        for key, value in qconfig.items():
            qconfig[key] = value.cuda()
    model_prepared = quantize_fx.prepare_fx(m.eval(), qconfig, example_input)
    # model_quantized = quantize_dynamic(model, qconfig_spec={nn.Conv2d}, dtype=torch.float16)
    if calibrate:
        co_transform_val = MyCoTransform(False, augment=False, height=512)
        dataset_val = cityscapes(args.datadir, co_transform_val, 'val')
        loader_val = DataLoader(dataset_val, num_workers=4, batch_size=args.batch_size, shuffle=False)
        print(len(loader_val))
        with torch.inference_mode():
            for step, (images, _) in enumerate(loader_val):
                print(f"step {step + 1}")
                print(images.shape)
                if not args.cpu:
                    images = images.cuda()
                images = Variable(images)
                model_prepared(images)
    model_quantized = quantize_fx.convert_fx(model_prepared)
    return model_quantized

def main(args, calibrate=True):
    savedir = f"{args.savedir}/quantized_model.pth"
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    model = Net(20)
    model = load_my_state_dict(model, torch.load(f"../trained_models/{args.model}_pretrained.pth", map_location=torch.device('cpu')))
    # for name, _  in model.state_dict().items():
    #     print(name)
    # flops, params = flopth(model, in_size=((3, 512, 1024),))
    # print(flops, params)
    total_params = sum(p.numel() for p in model.parameters() if torch.any(p != 0))
    print(total_params)

    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()
    qmodel = quantize_model2(model, args, calibrate)
    torch.save(qmodel.state_dict(), "./quantized_model.pth")
    return qmodel

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--model', default='erfnet')
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--savedir', default="/content/Drive/MyDrive/save")
    parser.add_argument('--datadir', default="/content/Drive/MyDrive/cityscapes")
    parser.add_argument('--resume', action='store_true')
    
    args = parser.parse_args()
    
    model = main(args, calibrate=True if args.resume else False)  # first execution initializes the model
    total_params = sum(p.numel() for p in model.parameters() if torch.any(p != 0))
    print(f"Total Params: {total_params}")  

    # dict = torch.load("quantized_model.pth", map_location=lambda storage, loc: storage)
    # # model = load_quant_dict(model, dict)
    # model.load_state_dict(dict)
    # print("dict loaded")
    file1 = open("file.txt", "w")
    for name, _ in model.state_dict().items():
        file1.write(f"{name}\n")
    # summary(model, input_size=(3, 512, 1024))
    # for p in model.parameters():
    #     print(p)
    # with profiler.profile(profile_memory=True, record_shapes=True) as prof:
    #     with profiler.record_function("model_inference"):
    #         model(torch.randn(1, 3, 512, 1024))
    # result = prof.key_averages().table(sort_by="self_cpu_time_total")
    # if "Flops" in result:
    #     print(f"Numero di FLOPS: {result["Flops"]}")
    # else:
    #     print("Statistiche dei FLOPS non disponibili nel profilo.")
    #     print(result)

    # flops, macs = get_model_complexity_info(model, (3, 512, 1024), as_strings=True, print_per_layer_stat=True)
    # print(f"FLOPS: {flops}")
    # print(f"Moltiplicazioni e addizioni (MACs): {macs}")

    flops, params = flopth(model, in_size=((3, 512, 1024),))
    print(flops, params)