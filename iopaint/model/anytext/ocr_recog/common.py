import torch
import torch.nn as nn
import torch.nn.functional as F

branch_coverage_swish = {
    "branch_11": False,  
    "branch_12": False   
}

branch_coverage_activation = {
    "branch_13": False,  
    "branch_14": False,   
    "branch_15": False,  
    "branch_16": False,   
    "branch_17": False,  
    "branch_18": False,   
    "branch_19": False, 
    "branch_20": False,
    "branch_41": False   
}

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

# out = max(0, min(1, slop*x+offset))
# paddle.fluid.layers.hard_sigmoid(x, slope=0.2, offset=0.5, name=None)
class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        # torch: F.relu6(x + 3., inplace=self.inplace) / 6.
        # paddle: F.relu6(1.2 * x + 3., inplace=self.inplace) / 6.
        return F.relu6(1.2 * x + 3., inplace=self.inplace) / 6.

class GELU(nn.Module):
    def __init__(self, inplace=True):
        super(GELU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.nn.functional.gelu(x)


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            branch_coverage_swish["branch_11"] = True
            x.mul_(torch.sigmoid(x))
            return x
        else:
            branch_coverage_swish["branch_12"] = True
            return x*torch.sigmoid(x)


class Activation(nn.Module):
    def __init__(self, act_type, inplace=True):
        super(Activation, self).__init__()
        act_type = act_type.lower()
        if act_type == 'relu':
            branch_coverage_activation["branch_13"] = True
            self.act = nn.ReLU(inplace=inplace)
        elif act_type == 'relu6':
            branch_coverage_activation["branch_14"] = True
            self.act = nn.ReLU6(inplace=inplace)
        elif act_type == 'sigmoid':
            branch_coverage_activation["branch_15"] = True
            raise NotImplementedError
        elif act_type == 'hard_sigmoid':
            branch_coverage_activation["branch_16"] = True
            self.act = Hsigmoid(inplace)
        elif act_type == 'hard_swish':
            branch_coverage_activation["branch_17"] = True
            self.act = Hswish(inplace=inplace)
        elif act_type == 'leakyrelu':
            branch_coverage_activation["branch_18"] = True
            self.act = nn.LeakyReLU(inplace=inplace)
        elif act_type == 'gelu':
            branch_coverage_activation["branch_19"] = True
            self.act = GELU(inplace=inplace)
        elif act_type == 'swish':
            branch_coverage_activation["branch_20"] = True
            self.act = Swish(inplace=inplace)
        else:
            branch_coverage_activation["branch_41"] = True #CHANGE LATER
            raise NotImplementedError

    def forward(self, inputs):
        return self.act(inputs)
    
# def print_coverage_swish():
#     for branch, hit in branch_coverage_swish.items():
#         print(f"{branch} in function Swish was {'hit' if hit else 'not hit'}")
#         branch_coverage_swish[branch] = False

# def print_coverage_activation():
#     for branch, hit in branch_coverage_activation.items():
#         print(f"{branch} in function Activation was {'hit' if hit else 'not hit'}")
#         branch_coverage_activation[branch] = False


# def main():
#     swish = Swish(inplace=True)
#     swish(torch.tensor([1.0, 2.0, 3.0]))
#     print("Test inplace branch(11)\n")
#     print_coverage_swish()
    
#     swish = Swish(inplace=False)
#     swish(torch.tensor([1.0, 2.0, 3.0]))
#     print("Test not-inplace branch(12)\n")
#     print_coverage_swish()
    
#     activation = Activation('relu')
#     activation(torch.tensor([1.0, 2.0, 3.0]))
#     print("Test relu branch(13)\n")
#     print_coverage_activation()
    
#     activation = Activation('relu6')
#     activation(torch.tensor([1.0, 2.0, 3.0]))
#     print("Test relu6 branch(14)\n")
#     print_coverage_activation()
    
#     activation = Activation('hard_sigmoid')
#     activation(torch.tensor([1.0, 2.0, 3.0]))
#     print("Test hard_sigmoid branch(16)\n")
#     print_coverage_activation()
    
#     activation = Activation('hard_swish')
#     activation(torch.tensor([1.0, 2.0, 3.0]))
#     print("Test hard_swish branch(17)\n")
#     print_coverage_activation()
    
#     activation = Activation('leakyrelu')
#     activation(torch.tensor([1.0, 2.0, 3.0]))
#     print("Test leakyrelu branch(18)\n")
#     print_coverage_activation()
    
#     activation = Activation('gelu')
#     activation(torch.tensor([1.0, 2.0, 3.0]))
#     print("Test gelu branch(19)\n")
#     print_coverage_activation()
    
#     activation = Activation('swish')
#     activation(torch.tensor([1.0, 2.0, 3.0]))
#     print("Test swish branch(20)\n")
#     print_coverage_activation()
    
#     try:
#         activation = Activation('sigmoid')
#     except NotImplementedError:
#         print("Test sigmoid branch(15)\n")
#         print_coverage_activation()
    
#     try:
#         activation = Activation('none')
#     except NotImplementedError:
#         print("Test not implemented branch(41)\n")
#         print_coverage_activation()

# if __name__ == '__main__':
#     main()