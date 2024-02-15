# import torch
# x = torch.randn(4, 20, 20, 768)
# # 重塑最后两维为 [4, 10, 10]，结果形状为 [4, 768, 4, 10, 10]
# # 这里我们首先将20x20的维度分解为4x10x10，然后调整张量形状以匹配这种分解
# x1 = x.view(4, 2, 10, 2, 10, 768).permute(0, 1, 3, 2, 4, 5).reshape(16, 100, 768)
# x2 = x1.view(4, 2, 2, 10, 10, 768).permute(0, 1, 3, 2, 4, 5).reshape(4, 20, 20, 768)
# import pdb;pdb.set_trace()
# print(x)


from utils.misc import SmoothedValue

sv = SmoothedValue()
import pdb;pdb.set_trace()
sv.update(1.0)
sv.update(2.0)
sv.update(3.0)
