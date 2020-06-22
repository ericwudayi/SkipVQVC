import torch
import torch.nn as nn
from gradient_penalty import gradient_penalty
def WGANGP(dis, voc, fake_voc):
    real = dis(voc)
    fake = dis(fake_voc)
    ld = (real - fake).mean()
    loss_dis = (ld +  0.01 * gradient_penalty(dis, 
        voc, fake_voc))
    
    loss_gan = fake.mean()

    return loss_dis, loss_gan