import torch
from torch import nn
from torch.nn import functional as F
from random import randint
class Quantize(nn.Module):
    def __init__(self, dim, n_embed):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        
        self.embedding = nn.Embedding(n_embed,dim,max_norm=1)
        
    def forward(self, input):
        embed = (self.embedding.weight.detach()).transpose(0,1)
        
        embed = (embed)/(torch.norm(embed,dim=0))
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
        
        return (quantize+quantize_1)/2, diff, embed_ind.detach().cpu()

class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor=2):
        super(PixelShuffle, self).__init__()
        # Custom Implementation because PyTorch PixelShuffle requires,
        # 4D input. Whereas, in this case we have have 3D array
        self.upscale_factor = upscale_factor

    def forward(self, input):
        n = input.shape[0]
        c_out = input.shape[1] // self.upscale_factor
        w_new = input.shape[2] * self.upscale_factor
        return input.view(n, c_out, w_new)

class Decoder(nn.Module):
    def __init__(
        self, in_channel, channel
    ):
        super().__init__()
        
        blocks = []
        mfd = 512
        num_groups = 4
        heads = []
        in_blocks = []
        self.scale = PixelShuffle(2)
        for i in range(1,4,1):
            
            block = GBlock(in_channel, mfd, channel, num_groups)
            
            blocks.append(block)
            
            head = nn.Conv1d(mfd, in_channel, 3, 1, 1, padding_mode='reflect')
            heads.append(head)
            in_blocks += [nn.GroupNorm(4, mfd)]
        
        self.blocks = nn.ModuleList(blocks)
        self.heads =  nn.ModuleList(heads[::-1])
        self.inblocks = nn.ModuleList(in_blocks)
        self.z_scale_factors = [2,2,2]

    def forward(self, q_after):
        q_after = q_after[::-1]
        x_body = 0#self.block0(q_after[0])
        x_head = 0#q_after[0] + self.head0(x_body) 
        output = []
        for i, (block, head, inb, scale_factor) in enumerate(zip(self.blocks, self.heads, self.inblocks, self.z_scale_factors)):
            code = F.interpolate(q_after[i], scale_factor=scale_factor, mode='nearest')
            if i != 0:
                x_body = F.interpolate(x_body, scale_factor=scale_factor, mode='nearest')
                x_head = F.interpolate(x_head, scale_factor=scale_factor, mode='nearest') 
            
            x_body = x_body + block(code)
            x_body = inb(x_body)
            x_head = x_head + head(x_body)

            output+=[x_head]
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
                nn.Conv1d(in_channel, channel, 4, stride=2, padding=1, padding_mode='reflect'),
                nn.LeakyReLU(),
                nn.Conv1d(channel, in_channel, 3, 1, 1,padding_mode='reflect'),
                
            ])]
        self.enc = nn.ModuleList(blocks)
        
        gnorm = []
        for i in range(3):
            gnorm += [nn.GroupNorm(2, in_channel)]
        self.gnorm = nn.ModuleList(gnorm)
        quantize_blocks = []
        
        for i in range(3):
            quantize_blocks += [
            Quantize(in_channel, n_embed)]
        self.quantize = nn.ModuleList(quantize_blocks)
        

        self.dec = Decoder(
            in_channel ,
            channel
        )
    def forward(self, input):
        enc_b, enc_input, diff, select_idx = self.encode(input)
        dec_1= self.decode(enc_b)
        
        return dec_1,  diff, select_idx

    def encode(self, input):
        x = input
        
        q_after_block = []
        enc_input = []
        diff_total = 0
        index_list = []

        for i, (enc_block, gn, quant) in enumerate(zip(self.enc, self.gnorm, self.quantize)):
            x = enc_block(x)   
            x = gn(x)

            x = x / torch.norm(x, dim = 1, keepdim = True)
            q_after, diff, select_idx = quant((x).permute(0,2,1))
            q_after = q_after.permute(0,2,1)
            #q_after = self.resample(q_after, 2**(2-i))
            enc_input += [x]
            q_after_block += [x]
            index_list += [select_idx]
            diff_total += diff
        
        return q_after_block, enc_input, diff_total, index_list

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
    def index_to_decode(self, index):
        q_a = []
        for i,l in enumerate(index):
            q_a += [self.quantize[i].embedding(l.detach().cuda().long()).transpose(1,2)]
        return self.decode(q_a)
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