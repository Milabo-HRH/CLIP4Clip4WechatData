import torch
import torchvision
from modules.until_module import align_state_dicts
state_dict = torch.load('./ckpts/ckpt_wechat_retrieval_looseType/pytorch_model.bin.0', map_location="cpu")
for k in list(state_dict.keys()):
    print(k)
# align_state_dicts(state_dict)
# lz = torch.jit.load('./modules/ViT-B-16.pt', map_location=torch.device('cpu'))
# for parameter in lz.state_dict():
#     if parameter not in state_dict.keys():
#         print("\""+parameter+"\",")
# print((lz.state_dict()['transformer.resblocks.3.attn.in_proj_weight']).shape)
# print((state_dict['module.bert.encoder.layer.3.attention.self.query.weight']).shape)
# print(torchvision.__version__)
# print(lz.state_dict()['positional_embedding'].shape)
# print(state_dict['module.bert.embeddings.position_embeddings.weight'])