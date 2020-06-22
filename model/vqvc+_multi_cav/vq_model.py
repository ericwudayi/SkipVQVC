import torch
from torch import nn
from torch.nn import functional as F

class Quantize(nn.Module):
    def __init__(self, dim, n_embed):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.dim = dim
        self.n_embed = n_embed
        data = torch.normal(torch.zeros(n_embed, dim), torch.ones(n_embed,dim)).float()
        
        self.embedding = torch.nn.Parameter(data)
        #self.embedding = nn.Embedding(n_embed,dim,max_norm=1)
    def forward(self, input):
        #print (self.embedding.pow(2).sum(1, keepdim=True).size())
        embed = self.embedding 
        embed = embed / (torch.norm(embed,dim=1, keepdim=True))
        dist_between = (
            embed.pow(2).sum(1, keepdim=True)
            - 2 * embed @ embed.transpose(0,1)
            + embed.pow(2).sum(1, keepdim=True).transpose(0,1)
        )
        #print (dist_between)
        entropy = torch.sum(dist_between)
        
        
        embed = self.embedding.detach().data.transpose(0, 1)
        embed = embed / (torch.norm(embed,dim=0))

        flatten = input.reshape(-1, self.dim).detach()
        '''
        print ("input size: ", flatten.size())
        print ("embed size: ", embed.size())
        print ("@size: ", (flatten @ embed).size())
        '''
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        quantize = self.embedding[embed_ind]/torch.norm(self.embedding[embed_ind],dim=1,keepdim=True)
        
        #embed_ind = embed_ind.view(*input.shape[:-1]).detach().cpu().cuda()
        quantize = quantize.view(*input.shape)
        #print (embed_ind.size())
        #print (quantize.size())
        #print (input.size())
        #print ("embed_ind size: ", embed_ind.size())
        diff = (quantize - input).pow(2).mean()
        quantize_1 = input + (quantize - input).detach()
        
        return (quantize+quantize_1)/2, diff - (1/self.n_embed)**2*entropy, embed_ind.detach().cpu()


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
            x = x + res(q_after[i]*std_embed[i] + sp_embed[i]) # x = x + res(q_after[i]*std_embed[i] +  sp_embed[i])
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
        
        speaker_blocks = []
        for i in range(3):
            speaker_blocks += [
            SBlock(in_channel//2**(i+1), channel, 8, 2**(2-i))
            ]
        self.speaker_blocks = nn.ModuleList(speaker_blocks)

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


        for i, (enc_block, quant, spk) in enumerate(zip(self.enc, self.quantize, self.speaker_blocks)):
            x = enc_block(x)   
            sp_embed, std_embed = spk(x)
            
            x_ = x - torch.mean(x, dim = 2, keepdim = True)
            std_ = torch.norm(x_, dim= 2, keepdim = True) + 1e-4
            x_ = x_ / std_

            q_after, diff, index= quant(x_.permute(0,2,1))
            q_after = q_after.permute(0,2,1)

            
            
            sp_embedding_block += [sp_embed]
            std_block += [std_embed]
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
            #RCBlock(mfd, ks, dilation=1, num_groups=num_groups),
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
    def forward(self, x):
        x = self.enc(x)
        x = torch.mean(x, dim=2)
        
        x =self.relu(self.dense(x))
        x1 = self.dense1(x)
        x2 = self.dense2(x)
        return x1.unsqueeze(2), x2.unsqueeze(2)