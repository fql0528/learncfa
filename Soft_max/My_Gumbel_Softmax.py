import torch.nn.functional as F
import  torch
import warnings
Tensor = torch.Tensor

#和原版的gumbel_softmax的区别就是把产生的gumble分布去掉了
# gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau) 改为了gumbels=logits/tau
def my_gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> Tensor:
    # if has_torch_function_unary(logits):
        # return handle_torch_function(gumbel_softmax, (logits,), logits, tau=tau, hard=hard, eps=eps, dim=dim)
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    # gumbels = (
        # -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    # )  # ~Gumbel(0,1)
    # gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    gumbels=logits*tau
    y_soft = gumbels.softmax(dim)
    # y_soft=F.softmax(gumbels,dim=1)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret