import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from transformers.models.t5.modeling_t5 import (
  T5Stack,
  T5Block,
  T5LayerSelfAttention,
  T5Attention,
  T5LayerNorm,
  T5LayerFF,
  T5DenseActDense,
  T5LayerCrossAttention,
)

import torch
import torch.nn as nn

import re

# TODO Tracing is missing (No graph used)

""" FLOP Calculation
  # https://openaccess.thecvf.com/content/ICCV2021/supplemental/Pan_Scalable_Vision_Transformers_ICCV_2021_supplemental.pdf

  if type(module) == nn.Linear:
    flops = 2 * module.in_features * module.out_features * batch 
  elif type(module) == T5LayerSelfAttention:
    # for kqv, flop = dim*dim*len(seq)
    # for attention map, flop = dim*len(seq)*len(seq)
    # for o, flop = len(seq)*len(seq)*dim
    # for concatenated self-attention outputs, flop = len(seq)*dim*dim
    # not sure the following is appropriate -- TODO double check
    flops = calculate_flops(module.SelfAttention.q) + calculate_flops(module.SelfAttention.k) 
            + calculate_flops(module.SelfAttention.v) + calculate_flops(module.SelfAttention.o) 
  elif type(module) == T5LayerFF:
    # mlp1: R^{d}-> R^{4d}, mlp2: R^{4d}-> R^{d}
  elif type(module) == T5DenseActDense:
    flops = calculate_flops(module.wi) + calculate_flops(module.wo)
  else:
    flops = 0
"""

def retrieve_flops(static_flops, name):
  if name in static_flops:
    return static_flops[name]
  else:
    0

def calculate_flops_wrapper(static_flops, module, name, batch=1):
  if type(module) == nn.Linear:
    flops = 2 * module.in_features * module.out_features * batch 
    static_flops[name] = flops
    static_flops['total'] += flops
  elif type(module) in [nn.ReLU, nn.Dropout]:
    static_flops[name] = 0
  elif type(module) == T5Attention:
    flops = 0
    for child in ['k', 'q', 'v', 'o']: # access module.k, module.q, module.v, module.o
      child_name = name+f'.{child}' 
      flops += retrieve_flops(static_flops, child_name)
    static_flops[name] = flops
  elif type(module) == T5LayerNorm:
    static_flops[name] = 0 
  elif type(module) == T5LayerSelfAttention:
    flops = retrieve_flops(static_flops, name+'.SelfAttention') # retrieve flops of module.SelfAttention
    static_flops[name] = flops
  elif type(module) == T5DenseActDense:
    flops = 0
    for child in ['wi', 'wo']:
      child_name = name+f'.{child}' 
      flops += retrieve_flops(static_flops, child_name)
    static_flops[name] = flops
  elif type(module) == T5LayerCrossAttention:
    flops = retrieve_flops(static_flops, name+'.EncDecAttention') # retrieve flops of module.SelfAttention
    static_flops[name] = flops
  elif type(module) in [T5LayerFF, T5Block]:
    # retrieve and sum children flops
    static_flops[name] = 0
  else:
    static_flops[name] = 0

  return 


tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
config = AutoConfig.from_pretrained("t5-small")


static_flops = {'total': 0}

# reverse traversal
for name, module in reversed(list(model.named_modules())):
  calculate_flops_wrapper(static_flops, module, name)

print(static_flops)
print(static_flops['total'])

num_gpu = torch.cuda.device_count()

assert num_gpu > 0, 'No GPU'

per_gpu_ideal = static_flops['total'] // num_gpu

# even_flop_cut
flops, last_child = 0, None
split_point = {}
for name, module in model.named_modules():
  if type(module) in [T5LayerSelfAttention, T5LayerFF, T5LayerCrossAttention]: ## coarse cut unit 
    flops += static_flops[name]
    if flops > per_gpu_ideal:
      # cut before this layer
      flops -= static_flops[name]
      assert last_child is not None, "Something wrong"
      split_point[last_child] = PipeSplitWrapper.SplitPoint.END 
    else:
      # add this to current module
      last_child = name
      continue
  else:
    last_child = name

#annotate_split_points(model, 
