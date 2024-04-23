# ERFNET full network definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn.utils.prune as prune


class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)
    

class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3,16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16,64))

        for x in range(0, 5):    #5 times
           self.layers.append(non_bottleneck_1d(64, 0.1, 1))  

        self.layers.append(DownsamplerBlock(64,128))

        for x in range(0, 2):    #2 times
            self.layers.append(non_bottleneck_1d(128, 0.1, 2))
            self.layers.append(non_bottleneck_1d(128, 0.1, 4))
            self.layers.append(non_bottleneck_1d(128, 0.1, 8))
            self.layers.append(non_bottleneck_1d(128, 0.1, 16))

        #only for encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128,64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64,16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d( 16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


class ERFNet(nn.Module):
    def __init__(self, num_classes, encoder=None):  #use encoder to pass pretrained encoder
        super().__init__()

        if (encoder == None):
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)    #predict=False by default
            return self.decoder.forward(output)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ERFNet(20).to(device = device)

def get_parameters_to_prune(module):
    parameters_to_prune = []
    for layer in module.children():
        if list(layer.children()):
            parameters_to_prune.extend(get_parameters_to_prune(layer))
        else:
            for name, param in layer.named_parameters():
              #print(name)
              if 'weight' in name:
                  #print(name)
                  parameters_to_prune.append((layer, param))
    return parameters_to_prune

parameters_to_prune = get_parameters_to_prune(model)
print(len(parameters_to_prune))
def count_layers_with_weights(module):
    if isinstance(module, nn.ModuleList):
        return sum(count_layers_with_weights(submodule) for submodule in module)
    elif isinstance(module, nn.Module) and any(param.requires_grad for param in module.parameters()):
        return 1 + sum(count_layers_with_weights(submodule) for submodule in module.children())
    else:
        return 0

# Conta il numero totale di layer con pesi nell'encoder
num_encoder_layers_with_weights = count_layers_with_weights(model.encoder)

# Conta il numero totale di layer con pesi nel decoder
num_decoder_layers_with_weights = count_layers_with_weights(model.decoder)


print("Numero di layer nell'encoder + decoder: ", num_encoder_layers_with_weights + num_decoder_layers_with_weights)
#pruning
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)
"""
#checking sparsity
def check_sparsity(module):
  for layer in module.children():
    if list(layer.children()):
            check_sparsity(layer)
    else:
          for name, param in layer.named_parameters():
                if 'weight' in name:
                  print("Sparsity in "+name+" : {:.2f}%".format(
        100. * float(torch.sum(param == 0))/ float(param.nelement())
                  ))
check_sparsity(model)

print(
    "Global sparsity: {:.2f}%".format(
        100. * float(
            torch.sum(model.conv1.weight == 0)
            + torch.sum(model.conv2.weight == 0)
            + torch.sum(model.fc1.weight == 0)
            + torch.sum(model.fc2.weight == 0)
            + torch.sum(model.fc3.weight == 0)
        )
        / float(
            model.conv1.weight.nelement()
            + model.conv2.weight.nelement()
            + model.fc1.weight.nelement()
            + model.fc2.weight.nelement()
            + model.fc3.weight.nelement()
        )
    )
)

module = model.encoder.output_conv
#unpruned model parameters
print(list(module.named_parameters()))
#pruned model parameters, removing at random 30% connections for one layer
prune.random_unstructured(module, name="weight", amount=0.3)
print(list(module.named_parameters()))
#mask generated and saved in buffer
print(list(module.named_buffers()))
# prune combines mask + original parameters to have the pruned version stored in weights
print(module.weight)
#pruning method applied before each forward pass with forward_pre_hooks, for each module pruned, it will acquire a pre hook associated with each parameter
print(module._forward_pre_hooks)
#we can prune bias too. For wxmple if we rune 3 smallest bias with l1 unstructure pruning
prune.l1_unstructured(module, name = "bias", amount = 3)
##the named parameters will be both weight orig and bias orig, whereas the buffer inclde weight mask and bias mask, the pruned version
##will exist as module attributes with two forward_pre hooks
print(list(module.named_parameters()))
"""
