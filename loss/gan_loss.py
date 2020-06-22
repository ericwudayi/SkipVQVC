import torch
import torch.nn as nn
from gradient_penalty import gradient_penalty
def GANLOSS(dis, voc, fake_voc):
    eps = 1e-4
    real = dis(voc)
    fake = dis(fake_voc)
    real = torch.clamp(real, eps, 1.0)
    fake = torch.clamp(fake, eps, 1.0)
    print ("real: ", real.mean())
    print ("fake: ", fake.mean())
    loss_dis = -(torch.log(real) + torch.log(1-fake)).mean()
     
    loss_gan = -torch.log(fake).mean()
    
    return loss_dis, loss_gan