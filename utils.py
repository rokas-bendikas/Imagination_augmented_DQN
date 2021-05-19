import torch as t
from device import Device


"""
def copy_gradients(target, source):
    for shared_param, param in zip(target.parameters(), source.parameters()):
        if param.grad is not None:
            shared_param._grad = param.grad.clone().cpu()


"""


def as_tensor(x, dtype=t.float32):
    return t.tensor(x, dtype=dtype, device=Device.get_device(),requires_grad=False)


    
