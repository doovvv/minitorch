from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    # 我们需要做的是让最后一个维度变成kh*kw，这样池化的时候可以直接对最后一个维度reduce
    tile_h = height / kh
    tile_w = width /kw
    input = input.contiguous()
    temp = input.view(batch,channel,tile_h,kh,tile_w,kw)
    temp = temp.permute(0,1,2,4,3,5)
    temp = temp.contiguous()
    ret = temp.view(batch,channel,tile_h,tile_w,kh*kw)
    return ret,tile_h,tile_w
    # raise NotImplementedError("Need to implement for Task 4.3")

def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    input,_,_ = tile(input,kernel)
    input = input.mean(len(input.shape)-1)
    ret = input.view(input.shape[0],input.shape[1],input.shape[2],input.shape[3])
    return ret
# TODO: Implement for Task 4.3.
