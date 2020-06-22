import torch
from torch import nn
from torch.nn import functional as F

class Quantize(nn.Module):
    def __init__(self, dim, n_embed):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.embedding = nn.Embedding(n_embed,dim,max_norm=1)
    def forward(self, input):
        embed = (self.embedding.weight.detach()).transpose(0,1)
        
        embed = (embed)/(torch.norm(embed,dim=0))
        flatten = input.reshape(-1, self.dim).detach()
        
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        _, embed_ind = (-dist).max(1)
        embed_ind = embed_ind.view(*input.shape[:-1]).detach().cpu().cuda()
        quantize = self.embedding(embed_ind)
        diff = (quantize - input).pow(2).mean()
        quantize_1 = input + (quantize - input).detach()
        
        return (quantize+quantize_1)/2, diff, embed_ind.detach().cpu()


class Decoder(nn.Module):
    def __init__(
        self, in_channel, channel
    ):
        super().__init__()
        
        blocks = []
        blocks_refine = []
        resblock = []
        num_groups = 4
        
        self.block0 = GBlock(in_channel//8, in_channel//8, channel, num_groups)
        
        for i in range(1,4,1):
            block = GBlock(in_channel//2**(i), in_channel//2**(i), channel, num_groups)
            blocks.append(block)
        for i in range(1,4,1):
            block = GBlock(in_channel//2**(i), in_channel//2**(i), channel, num_groups)
            blocks_refine.append(block)
        for i in range(1,4,1):
            block = GBlock(in_channel//2**(i), in_channel//2**(i), channel, num_groups)
            resblock += [block]
        
        self.blocks = nn.ModuleList(blocks[::-1])
        self.blocks_refine = nn.ModuleList(blocks_refine[::-1])
        self.resblock = nn.ModuleList(resblock[::-1])

        self.z_scale_factors = [2,2,2]

    def forward(self, q_after, sp_embed, std_embed):
        q_after = q_after[::-1]
        sp_embed = sp_embed[::-1]
        std_embed = std_embed[::-1] #not used, but it maybe is helpful
        output = []
        x = 0

        for i, (block, block_refine, res, scale_factor) in enumerate(zip(self.blocks, self.blocks_refine, self.resblock, self.z_scale_factors)):
            x = x + res(q_after[i] + sp_embed[i]) # x = x + res(q_after[i]*std_embed[i] +  sp_embed[i])
            x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
            x = x + block(x)
            x = torch.cat([x, x + block_refine(x)], dim = 1)
            output += [x]
        return output


class VC_MODEL(nn.Module):
    def __init__(
        self,
        in_channel=80,
        channel=512,
        n_embed = 128,
    ):
        super().__init__()

        blocks = []
        for i in range(3):
            blocks += [
            nn.Sequential(*[
                nn.Conv1d(in_channel//2**(i), channel, 4, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.Conv1d(channel, in_channel//2**(i+1), 3, 1, 1),
                
            ])]
        self.enc = nn.ModuleList(blocks)
        
        quantize_blocks = []
        
        for i in range(3):
            quantize_blocks += [
            Quantize(in_channel//2**(i+1), n_embed//2**i)]
        self.quantize = nn.ModuleList(quantize_blocks)
        
        
        self.dec = Decoder(
            in_channel ,
            channel
        )
    def forward(self, input):
        enc_b, sp_embed, std_block, diff, _ = self.encode(input)
        dec_1= self.decode(enc_b, sp_embed, std_block)
        idx = torch.randperm(enc_b[0].size(0))
        sp_shuffle = []
        std_shuffle = []
        for sm in (sp_embed):
            sp_shuffle += [sm[idx]]
        for std in std_block:
            std_shuffle += [std[idx]]
        
        dec_2 = self.decode(enc_b, sp_shuffle, std_shuffle)
        return dec_1, dec_2, enc_b, sp_embed, diff, idx

    def encode(self, input):
        x = input
        sp_embedding_block = []
        q_after_block = []
        std_block = []
        index_list = []
        diff_total = 0


        for i, (enc_block, quant) in enumerate(zip(self.enc, self.quantize)):
            x = enc_block(x)   
            x_ = x - torch.mean(x, dim = 2, keepdim = True)
            std_ = torch.norm(x_, dim= 2, keepdim = True) + 1e-4
            std_block += [std_]
            x_ = x_ / std_

            x_ = x_ / torch.norm(x_, dim = 1, keepdim = True)
            q_after, diff, index= quant(x_.permute(0,2,1))
            q_after = q_after.permute(0,2,1)
            
            sp_embed = torch.mean(x - q_after, 2, True)
            sp_embed = sp_embed / (torch.norm(sp_embed, dim = 1, keepdim=True)+1e-4) /3

            
            sp_embedding_block += [sp_embed]
            q_after_block += [q_after]
            diff_total += diff
            index_list += [index]
        return q_after_block, sp_embedding_block, std_block, diff_total, index_list


    def decode(self, quant_b, sp, std):
        
        dec_1 = self.dec(quant_b, sp, std)
        
        return dec_1
class RCBlock(nn.Module):
    def __init__(self, feat_dim, ks, dilation, num_groups):
        super().__init__()
        # ks = 3  # kernel size
        ksm1 = ks-1
        mfd = feat_dim
        di = dilation
        self.num_groups = num_groups

        self.relu = nn.LeakyReLU()

        self.rec = nn.GRU(mfd, mfd, num_layers=1, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(mfd, mfd, ks, 1, ksm1*di//2, dilation=di, groups=num_groups)
        self.gn = nn.GroupNorm(num_groups, mfd)

    def init_hidden(self, batch_size, hidden_size):
        num_layers = 1
        num_directions = 2
        hidden = torch.zeros(num_layers*num_directions, batch_size, hidden_size)
        hidden.normal_(0, 1)
        return hidden

    def forward(self, x):
        bs, mfd, nf = x.size()

        hidden = self.init_hidden(bs, mfd).to(x.device)

        r = x.transpose(1, 2)
        r, _ = self.rec(r, hidden)
        r = r.transpose(1, 2).view(bs, 2, mfd, nf).sum(1)
        c = self.relu(self.gn(self.conv(r)))
        x = x+r+c

        return x
class GBlock(nn.Module):
    def __init__(self, input_dim, output_dim, middle_dim, num_groups):
        super().__init__()

        ks = 3  # filter size
        mfd = middle_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mfd = mfd
        self.num_groups = num_groups

        # ### Main body ###
        block = [
            nn.Conv1d(input_dim, mfd, 3, 1, 1),
            nn.GroupNorm(num_groups, mfd),
            nn.LeakyReLU(),
            nn.Conv1d(mfd, output_dim, 3, 1, 1),
            
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):

        # ### Main ###
        x = self.block(x)

        return x
