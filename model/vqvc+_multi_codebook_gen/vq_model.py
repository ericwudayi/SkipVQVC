import torch
from torch import nn
from torch.nn import functional as F
from random import randint
class Quantize(nn.Module):
    def __init__(self, dim, n_embed):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        data = torch.normal(torch.zeros(n_embed, dim, dim), torch.ones(n_embed,dim, dim)).float()
        
        self.embedding = torch.nn.Parameter(data)
        
    def forward(self, input, speaker_embedding, pre_select_idx = None):
        
        speaker_embedding_mul = speaker_embedding[2].transpose(0,1)
        speaker_embedding_add = speaker_embedding[1].transpose(0,1)
        

        batch_size = speaker_embedding_mul.size(1)
        
        codebook = torch.matmul(self.embedding , speaker_embedding_mul)
        
        codebook = (codebook)/(torch.norm(codebook,dim=1,keepdim=True))
        
        codebook = codebook + speaker_embedding_add
        
        codebook = codebook.reshape(self.n_embed, self.dim, batch_size)
        

        embed = codebook.permute(2, 1, 0).detach()
        
        flatten = input.detach()
        
        dist = (
            flatten.pow(2).sum(2, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(1, keepdim=True)
        )
        _, embed_ind = (-dist).max(2)
        #print ("embed_ind size: ", embed_ind.size())
        embed = codebook.permute(2,0,1)
        quantize = []
        for i in range((embed.size(0))):
            quantize += [embed[i, embed_ind[i,:]]]
        quantize = torch.stack(quantize, dim = 0)
        
        diff = (quantize - input).pow(2).mean()
        quantize_1 = input + (quantize - input).detach()
        #print ("q_true", quantize_1.size())
        return (quantize+quantize_1)/2, diff , embed_ind

class QuantizeSimple(nn.Module):
    def __init__(self, dim, n_embed):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.embedding = nn.Embedding(n_embed,dim)
    def forward(self, input):
        embed = (self.embedding.weight.detach()).transpose(0,1)
        
        
        #input = input / torch.norm(input, dim = 2, keepdim=True)
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
        
        return (quantize+quantize_1)/2, diff


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

    def forward(self, q_after):
        q_after = q_after[::-1]
        x = 0
        output = []
        for i, (block, block_refine, res, scale_factor) in enumerate(zip(self.blocks, self.blocks_refine, self.resblock, self.z_scale_factors)):
            x = x + res(q_after[i])
            x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
            x = x + block(x)
            x = torch.cat([x, x + block_refine(x)], dim = 1)
            output+=[x]
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
        
        speaker_blocks = []
        for i in range(3):
            speaker_blocks += [
            SBlock(in_channel//2**(i+1), channel, 8, 2**(2-i))
            ]
        self.speaker_blocks = nn.ModuleList(speaker_blocks)

        quantize_blocks = []
        
        for i in range(3):
            quantize_blocks += [
            Quantize(in_channel//2**(i+1), n_embed//2**(2-i))]
        self.quantize = nn.ModuleList(quantize_blocks)
        
        quantize_simple_blocks = []

        for i in range(3):
            quantize_simple_blocks += [
            QuantizeSimple(in_channel//2**(i+1), 64)]
        self.quantize_simple = nn.ModuleList(quantize_simple_blocks)

        self.dec = Decoder(
            in_channel ,
            channel
        )
    def forward(self, input):
        enc_b, enc_input, sp_embedding_block, diff, select_idx = self.encode(input)
        dec_1= self.decode(enc_b)
        idx = list(torch.randperm(enc_b[0].size(0)).cpu().numpy())
        
        sp_shuffle = []
        for sm in (sp_embedding_block):
            sp_shuffle += [(sm[0][idx],sm[1][idx],sm[2][idx])]
        enc_b_change,  _,  select_idx2 = self.change_codebook(enc_input, sp_shuffle, select_idx)
        dec_2 = self.decode(enc_b_change)
        
        for i in range(3):
            diff += (enc_b[i] - enc_b_change[i]).mean()
        return dec_1, dec_2, enc_b, sp_embedding_block, diff, idx

    def encode(self, input):
        x = input
        sp_embedding_block = []
        q_after_block = []
        enc_input = []
        idx_block = []
        diff_total = 0


        for i, (enc_block, spk,quant, qs) in enumerate(zip(self.enc, self.speaker_blocks, self.quantize, self.quantize_simple)):
            x = enc_block(x)
            sp_embed = spk(x)
            
            x_ = x - torch.mean(x, dim = 2, keepdim = True)
            std_ = torch.norm(x_, dim= 2, keepdim = True) + 1e-4
            x_ = (x_ / std_)

            x_, diff_simple = qs(x_.permute(0,2,1))
            x_ = x_.permute(0,2,1)

            q_after, diff, select_idx = quant((x_+sp_embed[1].unsqueeze(2)).permute(0,2,1), sp_embed)
            q_after = q_after.permute(0,2,1)
            
            sp_embedding_block += [sp_embed]
            
            enc_input += [x_]
            q_after_block += [q_after]
            idx_block += [select_idx]

            diff_total += diff_simple
            diff_total += diff
        
        return q_after_block, enc_input, sp_embedding_block, diff_total, idx_block
    
    def resample(self, input, scale):
        input_resample = []#input
        
        for i in range(0,input.size(2), scale):
            
            mean_embedding = torch.mean(input[:,:,i:i+scale], dim=2, keepdim=True)
            select = randint(0,scale-1)
            input_resample += [input[:,:,i+select:i+select+1].detach() 
            + mean_embedding - mean_embedding.detach()]
        
        input_resample = torch.cat(input_resample,dim = 2)
        
        #input_resample = F.interpolate(input_resample, scale_factor=1/scale, mode='linear')
        input_resample = F.interpolate(input_resample, scale_factor=scale, mode='linear')
        return input_resample

    def change_codebook(self, input, speaker, select_idx):
        q_after_block = []
        diff_total = 0
        for i, (enc_input, quant, s, idx) in enumerate(zip(input,self.quantize, speaker, select_idx)):
            
            q_after, diff, select_idx = quant((enc_input +s[1].unsqueeze(2)).permute(0,2,1), s, idx)
            q_after = q_after.permute(0,2,1)
            q_after_block += [q_after]
            diff_total += diff
        return q_after_block, diff_total, select_idx
    def decode(self, quant_b):
        
        dec_1 = self.dec(quant_b)
        
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

        self.rec = nn.GRU(feat_dim, mfd, num_layers=1, batch_first=True, bidirectional=True)
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

class SBlock(nn.Module):
    def __init__(self, feat_dim, ks, dilation, num_groups):
        super().__init__()
        # ks = 3  # kernel size
        ksm1 = ks-1
        mfd = 128
        di = dilation
        self.feat_dim = feat_dim
        self.mfd = mfd
        self.num_groups = num_groups



        self.relu = nn.LeakyReLU()
        blocks =nn.Sequential(*[
                nn.Conv1d(feat_dim, mfd, ks, 1, ksm1*di//2, dilation=di, groups=num_groups),
                #nn.BatchNorm1d(mfd),
                nn.Conv1d(mfd, mfd, ks, 1, ksm1*di//2, dilation=di, groups=num_groups),
                nn.LeakyReLU(),
                nn.Conv1d(mfd, mfd, 3, 1, 1),
                #nn.BatchNorm1d(mfd),
            ])
        self.enc = blocks
        self.dense = nn.Linear(mfd,mfd)
        self.dense1 = nn.Linear(mfd,feat_dim)
        self.dense2 = nn.Linear(mfd,feat_dim)
        self.dense3 = nn.Linear(mfd,feat_dim)
    def forward(self, x):
        x = self.enc(x)
        x = torch.mean(x, dim=2)
        
        x =self.relu(self.dense(x))
        x1 = self.dense1(x)
        x2 = self.dense2(x)
        x3 = self.dense3(x)
        return x1, x2, x3


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
            #RCBlock(mfd, ks, dilation=1, num_groups=num_groups),
            nn.Conv1d(mfd, output_dim, 3, 1, 1),
            
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):

        # ### Main ###
        x = self.block(x)

        return x
