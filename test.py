import torch
import torchvision
from modules.until_module import align_state_dicts
state_dict = torch.load('./modules/clip_cn_vit-b-16.pt', map_location="cpu")['state_dict']
# for k in list(state_dict.keys()):
#     if k.startswith("module."):
#         state_dict[k[len("module."):]] = state_dict[k]
#         del state_dict[k]
# align_state_dicts(state_dict)
lz = torch.jit.load('./modules/ViT-B-16.pt', map_location=torch.device('cpu'))
# for parameter in lz.state_dict():
#     if parameter not in state_dict.keys():
#         print("\""+parameter+"\",")
print((lz.state_dict()['transformer.resblocks.3.attn.in_proj_weight']).shape)
print((state_dict['module.bert.encoder.layer.3.attention.self.query.weight']).shape)
print(torchvision.__version__)
print(lz.state_dict()['positional_embedding'].shape)
# print(state_dict['module.bert.embeddings.position_embeddings.weight'])