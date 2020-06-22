import torch
import torch.nn as nn
class BEGANRecorder(nn.Module):
    def __init__(self, lambda_k, init_k, gamma):
        super().__init__()
        self.lambda_k = lambda_k
        self.init_k = init_k
        self.gamma = gamma

        self.k = nn.Parameter(torch.tensor(init_k))

    def forward(self, real_dloss, fake_dloss, update_k=False):
        # convergence
        diff = self.gamma * real_dloss - fake_dloss
        convergence = (real_dloss + torch.abs(diff))

        # Update k
        if update_k:
            self.k.data = torch.clamp(self.k + self.lambda_k * diff, 0, 1).data

        return self.k.item(), convergence

def BEGANLoss(dis, voc, fake_voc, k, pad = 1):
    
    loss_gan  = torch.mean(torch.abs(dis(fake_voc)-fake_voc) * pad)
    real_dloss = torch.mean(torch.abs(dis(voc)-voc)*pad)
    fake_dloss = torch.mean(torch.abs(dis(fake_voc.detach())-fake_voc.detach())*pad)

    loss_dis = real_dloss - k * fake_dloss
    return loss_gan, loss_dis, real_dloss, fake_dloss

def BEGANLoss_v2(dis, voc, fake_voc):
    
    loss_gan  = torch.mean(torch.abs(dis(fake_voc)[0]-fake_voc))
    real_dloss = torch.mean(torch.abs(dis(voc)[0]-voc))
    fake_dloss = torch.mean(torch.abs(dis(fake_voc.detach())[0]-fake_voc.detach()))
    loss_dis = real_dloss + max(0, 0.5 - fake_dloss) + 0.01 * (dis(fake_voc)[1] + dis(voc)[1]).mean()
    
    return loss_gan, loss_dis, real_dloss, fake_dloss
