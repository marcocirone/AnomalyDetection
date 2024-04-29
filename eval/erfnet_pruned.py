# ERFNET full network definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.modules.module import Module
import torch.nn.utils.prune as prune
from torchsummary import summary
import pickle
import zipfile
import os

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


class Net(nn.Module):
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

model = Net(20).to(device = device)

"""
PRINT NUMERO TOTALE DI PARAMETRI PRE PRUNING
total_params = sum(p.numel() for p in model.parameters())
print("Numero totale di parametri nel modello:", total_params)

summary(model, (3, 512, 1024))
"""
def zip_model(input_file, output_file):
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(input_file)


torch.save(model, "model_before_pruning.pth")
zip_model("model_before_pruning.pth", "model_before_pruning.zip")

### GLOBAL PRUNING ####

## PREP TUPLE

def get_parameters_to_prune(module):
    parameters_to_prune = []
    for layer in module.children():
        if list(layer.children()):
            parameters_to_prune.extend(get_parameters_to_prune(layer))
        else:
            for name, param in layer.named_parameters():
              #print(name)
              if 'weight' in name or 'bias' in name:
                  #print(name)
                  parameters_to_prune.append((layer, name))
    return parameters_to_prune
                    

#parameters_to_prune = get_parameters_to_prune(model)
#print(len(parameters_to_prune))
"""
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
"""
"""
## GLOBAL PRUNING
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)
"""
#checking sparsity for every layer
def check_sparsity(module):
    for layer in module.children():
        if list(layer.children()):
            check_sparsity(layer)
        else:
            if hasattr(layer, 'weight'):
                weight = getattr(layer, 'weight')
                if weight is not None:
                    sparsity = 100. * float(torch.sum(weight == 0)) / float(weight.nelement())
                    print("Sparsity in {}: {:.2f}%".format(layer.__class__.__name__, sparsity))

#check_sparsity(model)


#checking global sparsity
def check_global_sparsity(model):
    total_zeros = 0
    total_elements = 0

    # Iterare su tutti i moduli della rete
    for module_name, module in model.named_modules():
        # Verificare se il modulo ha parametri 'weight'
        if hasattr(module, 'weight'):
            weight = getattr(module, 'weight')
            if weight is not None:
                # Aggiungere il numero di zeri nei pesi del modulo alla somma totale
                total_zeros += torch.sum(weight == 0).item()
                # Aggiungere il numero totale di elementi nei pesi del modulo alla somma totale
                total_elements += weight.nelement()

    global_sparsity = 100. * float(total_zeros) / float(total_elements)

    print("Global sparsity: {:.2f}%".format(global_sparsity))
  
#check_global_sparsity(model)
#summary(new_mod, (3, 512, 1024))
#total_params = sum(p.numel() for p in new_mod.parameters())
#print("Numero totale di parametri nel modello:", total_params)
## TOTAL AND NON ZERO PARAMETERS AFTER PRUNING
def count_and_print_weight(module):
    total_params = 0
    non_zero_params_total = 0

    for layer in module.children():
        if list(layer.children()):
            count, non_zero_count = count_and_print_weight(layer)
            total_params += count
            non_zero_params_total += non_zero_count
        else:
            if hasattr(layer, 'weight'):
                weights = layer.weight
                total_params += weights.numel()
                non_zero_params = weights[weights != 0]
                non_zero_params_total += non_zero_params.numel()

                

    return total_params, non_zero_params_total

#tot, nonzero = count_and_print_weight(model) 
#print("Total parameters: ", tot,"\nNon zero parameters: ", nonzero)

## POST PRUNING

def remove_pruned_weights(module):
    trainable_params = []
    for layer in module.children():
        if list(layer.children()):
            remove_pruned_weights(layer)
        else:
            if hasattr(layer, 'weight') and layer.weight.requires_grad:
                prune.remove(layer, 'weight')  # Rimuovi completamente i pesi prunati
                #print("rimosso")
                #print(layer.weight)
                

#remove_pruned_weights(model)
#summary(model, (3, 512, 1024))
"""
torch.save(model, "model_after_pruning.pth")
zip_model("model_after_pruning.pth", "model_after_pruning.zip")

size_before_pruning = os.path.getsize("model_before_pruning.zip")
size_after_pruning = os.path.getsize("model_after_pruning.zip")

# Comparing model size pre and post pruning

print("Zipped model size before pruning:", size_before_pruning/(1024*1024), "MB")
print("Zipper model size after pruning:", size_after_pruning/(1024*1024), "MB")
"""
def prune_and_return_model(model, pruning_amount):
    # Pruning
    parameters_to_prune = get_parameters_to_prune(model)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_amount,
    )


    tot, nonzero = count_and_print_weight(model) 
    print("Total parameters: ", tot,"\nNon zero parameters: ", nonzero)
    #Rimozione dei pesi pruned
    #remove_pruned_weights(model)

    # Salvataggio del modello pruned
    #torch.save(model, "model_after_pruning.pth")

    # Zippare il modello pruned
    #zip_model("model_after_pruning.pth", "model_after_pruning.zip")


    """
    # Calcolo delle dimensioni del modello zippato prima e dopo il pruning
    size_before_pruning = os.path.getsize("model_before_pruning.zip")
    size_after_pruning = os.path.getsize("model_after_pruning.zip")

    # Stampa delle dimensioni dei modelli zippati prima e dopo il pruning
    print("Zipped model size before pruning:", size_before_pruning / (1024 * 1024), "MB")
    print("Zipped model size after pruning:", size_after_pruning / (1024 * 1024), "MB")
    """
    return model

def remove_and_save(model):
  #Rimozione dei pesi pruned
    remove_pruned_weights(model)

    # Salvataggio del modello pruned
    torch.save(model, "model_after_pruning.pth")

    # Zippare il modello pruned
    zip_model("model_after_pruning.pth", "model_after_pruning.zip")
    return model

#new_mod = prune_and_return_model(model, 0.3)
"""
### LOCAL PRUNING ###

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