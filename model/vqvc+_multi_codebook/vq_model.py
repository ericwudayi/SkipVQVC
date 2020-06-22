import torch
from torch import nn
from torch.nn import functional as F
from random import randint
class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.embedding = []
        for i in range(2):
            self.embedding += [nn.Embedding(n_embed,dim)]
        self.embedding = nn.ModuleList(self.embedding)
    def forward(self, input, input_code_select):
        
        embed_ind_select = []
        for name in (input_code_select):
            embed_ind_select += [int(name[-1])-5]
        
        embed_ind_select = torch.tensor(embed_ind_select).long().cuda()
        
        embedding_list = []
        for i in range(embed_ind_select.size(0)):
            embed_layer_select = self.embedding[embed_ind_select[i]]
            embed_layer_select_weight = embed_layer_select.weight.detach().transpose(0,1)
            embedding_list += [ embed_layer_select_weight]
        
        embed = torch.stack(embedding_list,dim=0)
        embed = (embed)/(torch.norm(embed,dim=0))
        flatten = input.detach()
        dist = (
            flatten.pow(2).sum(2, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(1, keepdim=True)
        )
        
        _, embed_ind = (-dist).max(2)
        
        quantize = []
        for i in range(embed_ind.size(0)):
            quantize += [ self.embedding[embed_ind_select[i]](embed_ind[i].unsqueeze(0)) ]
        quantize = torch.cat(quantize,dim=0)
        diff = (quantize - input).pow(2).mean()
        quantize_1 = input + (quantize - input).detach()
        
        return (quantize+quantize_1)/2, diff , embed_ind_select


class Decoder(nn.Module):
    def __init__(
        self, in_channel, channel
    ):
        super().__init__()
        
        blocks = []
        blocks_refine = []
        resblock = []
        num_groups = 4
                
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
            Quantize(in_channel//2**(i+1), n_embed)]
        self.quantize = nn.ModuleList(quantize_blocks)
        

        self.dec = Decoder(
            in_channel ,
            channel
        )
    def forward(self, input, name):
        enc_b, enc_input, diff, select_idx = self.encode(input, name)
        dec_1= self.decode(enc_b)
        
        name_shuffle = []
        
        for i in range(len(name)):
            if name[i] == 'p225':
                name_shuffle += ['p226']
            else:
                name_shuffle += ['p225']
            
        enc_b_change,select_idx2 = self.change_codebook(enc_input, name_shuffle)
        
        dec_2 = self.decode(enc_b_change)
        return dec_1, dec_2, enc_b, diff

    def encode(self, input, name):
        x = input
        
        q_after_block = []
        enc_input = []
        diff_total = 0


        for i, (enc_block, spk, quant) in enumerate(zip(self.enc, self.speaker_blocks, self.quantize)):
            x = enc_block(x)   
            
            #x_ = x - torch.mean(x, dim = 2, keepdim = True)
            #x_ = x_ / torch.norm(x_, dim= 2, keepdim = True) + 1e-4

            x = x / torch.norm(x, dim = 1, keepdim = True)
            q_after, diff, select_idx = quant( (x).permute(0,2,1), name)
            q_after = q_after.permute(0,2,1)
            #q_after = self.resample(q_after, 2**(2-i))
            enc_input += [x]
            q_after_block += [q_after]
            diff_total += diff
        
        return q_after_block, enc_input, diff_total, select_idx

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

    def change_codebook(self, input, name):
        q_after_block = []
        for i, (enc_input, quant) in enumerate(zip(input, self.quantize)):
            q_after, _, select_idx = quant((enc_input).permute(0,2,1), name)
            q_after = q_after.permute(0,2,1)
            q_after_block += [q_after]

        return q_after_block, select_idx
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
            RCBlock(mfd, ks, dilation=1, num_groups=num_groups),
            nn.Conv1d(mfd, output_dim, 3, 1, 1),
            
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):

        # ### Main ###
        x = self.block(x)

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
                nn.Conv1d(mfd, mfd, ks, 1, ksm1*di//2, dilation=di, groups=num_groups),
                nn.LeakyReLU(),
                nn.Conv1d(mfd, mfd, 3, 1, 1),
            ])
        self.enc = blocks
        self.dense = nn.Linear(mfd,mfd)
        self.dense1 = nn.Linear(mfd,feat_dim)
        self.dense2 = nn.Linear(mfd,feat_dim)
    def forward(self, x):
        x = self.enc(x)
        x = torch.mean(x, dim=2)
        
        x =self.relu(self.dense(x))
        x1 = self.dense1(x)
        x2 = self.dense2(x)
        return x1, x2