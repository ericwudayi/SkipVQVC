import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

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

class BNSNConv1dDBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dilation):
        super().__init__()
        ks = kernel_size
        ksm1d2 = (ks - 1) // 2

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.dilation = dilation

        block = [
            spectral_norm(nn.Conv1d(input_dim, output_dim, ks,
                                    1, dilation*ksm1d2, dilation=dilation)),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):

        x = self.block(x)

        return x

class BNSNConv2dDBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, frequency_stride, time_dilation):
        super().__init__()
        ks = kernel_size
        ksm1d2 = (ks - 1) // 2

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.time_dilation = time_dilation
        self.frequency_stride = frequency_stride

        block = [
            spectral_norm(nn.Conv2d(
                input_dim, output_dim, ks,
                (frequency_stride, 1),
                (1, time_dilation*ksm1d2),
                dilation=(1, time_dilation))),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):

        x = self.block(x)
        
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


class NetD(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        ks = 3  # filter size
        mfd = 512

        self.mfd = mfd
        self.input_size = input_size

        # ### Main body ###
        blocks2d = [
            BNSNConv2dDBlock(1, 4, ks, 2, 2),
            BNSNConv2dDBlock(4, 16, ks, 2, 4),
            BNSNConv2dDBlock(16, 64, ks, 2, 8),
            #BNSNConv2dDBlock(64, 128, ks, 2, 16)
        ]

        blocks1d = [
            BNSNConv1dDBlock(64*10 * input_size//80, mfd, 3, 1),
            BNSNConv1dDBlock(mfd, mfd, ks, 16),
            BNSNConv1dDBlock(mfd, mfd, ks, 32),
            BNSNConv1dDBlock(mfd, mfd, ks, 64),
            BNSNConv1dDBlock(mfd, mfd, ks, 128),
            #BNSNConv1dDBlock(mfd, mfd, ks, 256),
        ]

        self.body2d = nn.Sequential(*blocks2d)
        self.body1d = nn.Sequential(*blocks1d)

        self.head = spectral_norm(nn.Conv1d(mfd, input_size, 3, 1, 1))

    def forward(self, x):
        '''
        x.shape=(batch_size, feat_dim, num_frames)
        cond.shape=(batch_size, cond_dim, num_frames)
        '''
        bs, fd, nf = x.size()

        # ### Process generated ###
        # shape=(bs, 1, fd, nf)
        x = x.unsqueeze(1)

        # shape=(bs, 64, 10, nf_)
        x = self.body2d(x)
        # shape=(bs, 64*10, nf_)
        x = x.view(bs, -1, x.size(3))

        # ### Merging ###
        x = self.body1d(x)

        # ### Head ###
        # shape=(bs, input_size, nf)
        # out = torch.sigmoid(self.head(x))
        out = self.head(x)

        return out

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.embedding = nn.Embedding(n_embed,dim)
        self.inorm = nn.InstanceNorm1d(dim)
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

    def forward(self, q_after, sp_embed, std_embed):
        q_after = q_after[::-1]
        sp_embed = sp_embed[::-1]
        std_embed = std_embed[::-1]
        x = 0

        for i, (block, block_refine, res, scale_factor) in enumerate(zip(self.blocks, self.blocks_refine, self.resblock, self.z_scale_factors)):
            x = x + res(q_after[i]*sp_embed[i] + sp_embed[i])
            x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
            x = x + block(x)
            x = torch.cat([x, x + block_refine(x)], dim = 1)
        return x


    
class VQVAE(nn.Module):
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
            Quantize(in_channel//2**(i+1), n_embed//2**(2-i))]
        self.quantize = nn.ModuleList(quantize_blocks)
        
        
        self.dec = Decoder(
            in_channel ,
            channel
        )
    def forward(self, input):
        enc_b, sp_embed, std_block, diff = self.encode(input)
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
        diff_total = 0


        for i, (enc_block, quant) in enumerate(zip(self.enc, self.quantize)):
            x = enc_block(x)   
            x_ = x - torch.mean(x, dim = 2, keepdim = True)
            std_ = torch.norm(x_, dim= 2, keepdim = True) + 1e-4
            std_block += [std_]
            x_ = x_ / std_

            x_ = x_ / torch.norm(x_, dim = 1, keepdim = True)
            q_after, diff = quant(x_.permute(0,2,1))
            q_after = q_after.permute(0,2,1)
            
            sp_embed = torch.mean(x - q_after, 2, True)
            sp_embed = sp_embed / (torch.norm(sp_embed, dim = 1, keepdim=True)+1e-4) /3

            sp_embedding_block += [sp_embed]
            q_after_block += [q_after]
            diff_total += diff
        
        return q_after_block, sp_embedding_block, std_block, diff_total

    def decode(self, quant_b, sp, std):
        
        dec_1 = self.dec(quant_b, sp, std)
        
        return dec_1


class Decoder_VAE(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()
        
        blocks = []
        blocks_refine = []
        resblock = []
        num_groups = 4
        
        self.block0 = GBlock(in_channel//8, in_channel//8, channel, num_groups)
        if stride == 8:
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
        std_embed = std_embed[::-1]
        x = 0
        for i, (block, block_refine, res, scale_factor) in enumerate(zip(self.blocks, self.blocks_refine, self.resblock, self.z_scale_factors)):
            x = x + res(q_after[i] + sp_embed[i])
            x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
            x = x + block(x)
            x = torch.cat([x, x + block_refine(x)], dim = 1)
        return x

class VQVAE_VAE(nn.Module):
    def __init__(
        self,
        in_channel=80,
        channel=512,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
        embed_pre = 256
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
            Quantize(in_channel//2**(i+1), n_embed//2**(2-i))]
        self.quantize = nn.ModuleList(quantize_blocks)
        
        
        self.dec = Decoder_VAE(
            in_channel ,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=8,
        )
    def forward(self, input):
        enc_b, sp_embed, std_block,  diff = self.encode(input)
        #print (enc_speaker.size(), enc_b.size(), enc_b_content.size())
        dec_1= self.decode(enc_b, sp_embed, std_block)
        idx = torch.randperm(enc_b[0].size(0))
        sp_shuffle = []
        std_shuffle = []
        bias_shuffle = []
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
        diff_total = 0


        for i, (enc_block, quant) in enumerate(zip(self.enc, self.quantize)):
            x = enc_block(x)   
            
            x_ = x - torch.mean(x, dim = 2, keepdim = True)
            std_ = torch.norm(x_, dim= 2, keepdim = True) + 1e-4
            std_block += [std_]
            x_ = x_ / std_

            x_ = x_ / torch.norm(x_, dim = 1, keepdim = True)
            q_after, diff = quant(x_.permute(0,2,1))
            q_after = q_after.permute(0,2,1)
            
            sp_embed = torch.mean(x - q_after, 2)
            sp_std = torch.std(x - q_after, 2)

            kl_div = -0.5*(sp_std**2+1-sp_embed**2-torch.exp(sp_std**2))
            
            sp_embed = sp_embed + torch.randn(sp_std.size()).cuda()*sp_std
            sp_embedding_block += [sp_embed.unsqueeze(2)]
            
            q_after_block += [q_after]
            diff_total += diff
            diff_total += torch.mean(kl_div)
        return q_after_block, sp_embedding_block, std_block, diff_total#, x__, x___

    def decode(self, quant_b, sp, std):
        
        dec_1 = self.dec(quant_b, sp, std)
        
        return dec_1

class VQVAE_RESAMPLE(nn.Module):
    def __init__(
        self,
        in_channel=80,
        channel=512,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
        embed_pre = 256
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
        quantize_speakers = []
        for i in range(3):
            quantize_blocks += [
            Quantize(in_channel//2**(i+1), n_embed//2**(2-i))]
        self.quantize = nn.ModuleList(quantize_blocks)
        for i in range(3):
            quantize_speakers += [
            Quantize(in_channel//2**(i+1), n_embed//2**(2-i))]
        self.quantize = nn.ModuleList(quantize_blocks)
        self.quantize_speakers = nn.ModuleList(quantize_speakers)
        
        self.dec = Decoder(
            in_channel ,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=8,
        )
    def forward(self, input):
        enc_b, sp_embed, std_block, diff = self.encode(input)
        
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
        diff_total = 0


        for i, (enc_block, quant, quantize_sp) in enumerate(zip(self.enc, self.quantize, self.quantize_speakers)):
            x = enc_block(x)   
            
            x_ = x - torch.mean(x, dim = 2, keepdim = True)
            std_ = torch.norm(x_, dim= 2, keepdim = True) + 1e-4
            std_block += [std_]
            x_ = x_ / std_

            x_ = x_ / torch.norm(x_, dim = 1, keepdim = True)
            q_after, diff = quant(x_.permute(0,2,1))
            q_after = q_after.permute(0,2,1)
            
            sp_embed = torch.mean(x - q_after, 2, True)
            sp_embed = sp_embed / (torch.norm(sp_embed, dim = 1, keepdim=True)+1e-4) /3

            sp_embed, diff_speaker = quantize_sp(sp_embed.permute(0,2,1))
            sp_embed = sp_embed.permute(0,2,1)

            q_after = self.resample(q_after)
            sp_embedding_block += [sp_embed]
            q_after_block += [q_after]
            diff_total += diff
            diff_total += diff_speaker
        return q_after_block, sp_embedding_block, std_block, diff_total


    def resample(self, input):
        #input = F.interpolate(input, scale_factor=0.5, mode='nearest')
        #input = F.interpolate(input, scale_factor=2, mode='nearest')
        #input = F.interpolate(input, scale_factor=1/4, mode='linear')
        #input = F.interpolate(input, scale_factor=4, mode='nearest')
        input = F.interpolate(input, scale_factor=1/8, mode='linear')
        input = F.interpolate(input, scale_factor=8, mode='nearest')

        return input
    
    def encode(self, input):
        x = input
        sp_embedding_block = []
        q_after_block = []
        std_block = []
        diff_total = 0


        for i, (enc_block, quant) in enumerate(zip(self.enc, self.quantize)):
            x = enc_block(x)   
            
            x_ = x - torch.mean(x, dim = 2, keepdim = True)
            std_ = torch.norm(x_, dim= 2, keepdim = True) + 1e-4
            std_block += [std_]
            x_ = x_ / std_

            x_ = x_ / torch.norm(x_, dim = 1, keepdim = True)
            q_after, diff = quant(x_.permute(0,2,1))
            q_after = q_after.permute(0,2,1)
            
            sp_embed = torch.mean(x_ - q_after, 2, True)
            sp_embed = sp_embed / (torch.norm(sp_embed, dim = 1, keepdim=True)+1e-4) /3

            sp_embedding_block += [sp_embed]
            q_after_block += [q_after]
            diff_total += diff
        
        return q_after_block, sp_embedding_block, std_block, diff_total#, x__, x___

    def decode(self, quant_b, sp, std):
        
        dec_1 = self.dec(quant_b, sp, std)
        
        return dec_1
class VQVAE_SPEAKER(nn.Module):
    def __init__(
        self,
        in_channel=80,
        channel=512,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
        embed_pre = 256
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
        quantize_speakers = []
        for i in range(3):
            quantize_blocks += [
            Quantize(in_channel//2**(i+1), n_embed//2**(2-i))]
        
        for i in range(3):
            quantize_speakers += [
            Quantize(in_channel//2**(i+1), 8**i)]
        self.quantize = nn.ModuleList(quantize_blocks)
        self.quantize_speakers = nn.ModuleList(quantize_speakers)
        
        self.dec = Decoder(
            in_channel ,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=8,
        )
    def forward(self, input):
        enc_b, sp_embed, std_block, diff = self.encode(input)
        #print (enc_speaker.size(), enc_b.size(), enc_b_content.size())
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
        diff_total = 0


        for i, (enc_block, quant, quantize_sp) in enumerate(zip(self.enc, self.quantize, self.quantize_speakers)):
            x = enc_block(x)   
            
            x_ = x - torch.mean(x, dim = 2, keepdim = True)
            std_ = torch.norm(x_, dim= 2, keepdim = True) + 1e-4
            std_block += [std_]
            x_ = x_ / std_

            x_ = x_ / torch.norm(x_, dim = 1, keepdim = True)
            q_after, diff = quant(x_.permute(0,2,1))
            q_after = q_after.permute(0,2,1)
            
            sp_embed = torch.mean(x - q_after, 2, True)
            sp_embed = sp_embed / (torch.norm(sp_embed, dim = 1, keepdim=True)+1e-4) /3

            sp_embed, diff_speaker = quantize_sp(sp_embed.permute(0,2,1))
            sp_embed = sp_embed.permute(0,2,1)

            sp_embedding_block += [sp_embed]
            q_after_block += [q_after]
            diff_total += diff
            diff_total += diff_speaker
        return q_after_block, sp_embedding_block, std_block, diff_total

    def decode(self, quant_b, sp, std):
        
        dec_1 = self.dec(quant_b, sp, std)
        
        return dec_1

class VQVAE_V2(nn.Module):
    def __init__(
        self,
        in_channel=80,
        channel=512,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
        embed_pre = 256
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
            Quantize(in_channel//2**(i+1), n_embed//2**(2-i))]
        self.quantize = nn.ModuleList(quantize_blocks)
        
        
        self.dec = Decoder(
            in_channel ,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=8,
        )
    def forward(self, input):
        enc_b, sp_embed, std_block, diff = self.encode(input)
        #print (enc_speaker.size(), enc_b.size(), enc_b_content.size())
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
        diff_total = 0


        for i, (enc_block, quant) in enumerate(zip(self.enc, self.quantize)):
            x = enc_block(x)   
            
            x_ = x - torch.mean(x, dim = 2, keepdim = True)
            std_ = torch.norm(x_, dim= 2, keepdim = True) + 1e-4
            std_block += [std_]
            x_ = x_ / std_

            x_ = x_ / torch.norm(x_, dim = 1, keepdim = True)
            q_after, diff = quant(x_.permute(0,2,1))
            q_after = q_after.permute(0,2,1)
            
            sp_embed = torch.mean(x_ - q_after, 2, True)
            sp_embed = sp_embed / (torch.norm(sp_embed, dim = 1, keepdim=True)+1e-4) /3

            sp_embedding_block += [sp_embed]
            q_after_block += [q_after]
            diff_total += diff
        
        return q_after_block, sp_embedding_block, std_block, diff_total#, x__, x___

    def decode(self, quant_b, sp, std):
        
        dec_1 = self.dec(quant_b, sp, std)
        
        return dec_1

class Decoder_RHYTHM(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()
        
        blocks = []
        blocks_refine = []
        resblock = []
        num_groups = 4
        
        self.block0 = GBlock(in_channel//8, in_channel//8, channel, num_groups)
        if stride == 8:
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

    def forward(self, q_after, sp_embed, std_embed, rhythm_embed):
        q_after = q_after[::-1]
        sp_embed = sp_embed[::-1]
        std_embed = std_embed[::-1]
        rhythm_embed = rhythm_embed[::-1]
        x = q_after[0]
        
        x = self.block0(x * std_embed[0] + sp_embed[0] + rhythm_embed[0])
        for i, (block, block_refine, res, scale_factor) in enumerate(zip(self.blocks, self.blocks_refine, self.resblock, self.z_scale_factors)):
            q_after[i] = F.interpolate(q_after[i], scale_factor=scale_factor, mode='nearest')
            rhythm_embed[i] = F.interpolate(rhythm_embed[i], scale_factor=scale_factor, mode='nearest')
            x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
            x = x + res(q_after[i]*std_embed[i] + sp_embed[i] )
            x = x + block(x)
            x = torch.cat([x, x + block_refine(x)], dim = 1)
            #x = torch.stack([x, x+block_refine(x)], dim = 1).view(x.size(0), 2 * x.size(1), -1)  
        return x

class VQVAE_RHYTHM(nn.Module):
    def __init__(
        self,
        in_channel=80,
        channel=512,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
        embed_pre = 256
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
            Quantize(in_channel//2**(i+1), n_embed)]
        self.quantize = nn.ModuleList(quantize_blocks)
        
        
        self.dec = Decoder_RHYTHM(
            in_channel ,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=8,
        )
    def forward(self, input_rhy, input):
        enc_b, sp_embed, std_block, rhy_block, diff = self.encode(input_rhy, input)
        #print (enc_speaker.size(), enc_b.size(), enc_b_content.size())
        dec_1= self.decode(enc_b, sp_embed, std_block, rhy_block)


        #enc_b2, sp_embed2, std_block2, rhy_block2, diff2 = self.encode(input, input_rhy)
        #print (enc_speaker.size(), enc_b.size(), enc_b_content.size())
        #dec_2= self.decode(enc_b2, sp_embed2, std_block2, rhy_block2)
        idx = torch.randperm(enc_b[0].size(0))
        
        sp_shuffle = []
        std_shuffle = []
        rhy_shuffle = []
        
        for sm in (sp_embed):
            sp_shuffle += [sm[idx]]
        for std in std_block:
            std_shuffle += [std[idx]]
        for rhy in rhy_block:
            rhy_shuffle += [rhy[idx]]

        dec_2 = self.decode(enc_b, sp_shuffle, std_shuffle, rhy_shuffle)
        
        
        return dec_1, dec_2, enc_b, sp_embed, diff, idx
    def resample(self, input):
        #input = F.interpolate(input, scale_factor=0.5, mode='linear')
        #input = F.interpolate(input, scale_factor=2, mode='nearest')
        #input = F.interpolate(input, scale_factor=1/4, mode='linear')
        #input = F.interpolate(input, scale_factor=4, mode='nearest')
        #input = F.interpolate(input, scale_factor=1/8, mode='linear')
        #input = F.interpolate(input, scale_factor=8, mode='nearest')

        return input
    def encode(self, input_rhy, input):
        x = input
        x_rhy = input_rhy
        sp_embedding_block = []
        q_after_block = []
        rhythm_block = []
        std_block = []
        diff_total = 0


        for i, (enc_block, quant) in enumerate(zip(self.enc, self.quantize)):
            x = enc_block(x)   
            x_rhy = enc_block(x_rhy)

            rhy = torch.mean(x_rhy, dim = 1, keepdim = True)
            
            x_ = x - torch.mean(x, dim = 2, keepdim = True)
            std_ = torch.norm(x_, dim= 2, keepdim = True) + 1e-4
            std_block += [std_]
            x_ = x_ / std_
            
            x_ = x_ / torch.norm(x_, dim = 1, keepdim = True)
            q_after, diff = quant(x_.permute(0,2,1))
            q_after = q_after.permute(0,2,1)
            
            sp_embed = torch.mean(x - q_after, 2, True)
            sp_embed = sp_embed / (torch.norm(sp_embed, dim = 1, keepdim=True)+1e-4) /3

            q_after = self.resample(q_after)
            q_after = q_after - torch.mean(q_after, dim= 1 , keepdim = True)
            sp_embedding_block += [sp_embed]
            q_after_block += [q_after]
            rhythm_block += [rhy]
            diff_total += diff
        
        return q_after_block, sp_embedding_block, std_block, rhythm_block, diff_total 

    def decode(self, quant_b, sp, std, rhy):
        
        dec_1 = self.dec(quant_b, sp, std, rhy)
        
        return dec_1



class Decoder_PITCH(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()
        
        blocks = []
        blocks_refine = []
        resblock = []
        num_groups = 4
        
        self.block0 = GBlock(in_channel//8, in_channel//8, channel, num_groups)
        if stride == 8:
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

    def forward(self, q_after, sp_embed, std_embed, pitch_embed):
        q_after = q_after[::-1]
        sp_embed = sp_embed[::-1]
        std_embed = std_embed[::-1]
        pitch_embed = pitch_embed[::-1]
        x = q_after[0]
        
        x = self.block0(x * std_embed[0] + sp_embed[0] + pitch_embed[0])
        for i, (block, block_refine, res, scale_factor) in enumerate(zip(self.blocks, self.blocks_refine, self.resblock, self.z_scale_factors)):
            q_after[i] = F.interpolate(q_after[i], scale_factor=scale_factor, mode='nearest')
            
            x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
            x = x + res(q_after[i]*std_embed[i] + sp_embed[i] + pitch_embed[i] )
            x = x + block(x)
            x = torch.cat([x, x + block_refine(x)], dim = 1)
            #x = torch.stack([x, x+block_refine(x)], dim = 1).view(x.size(0), 2 * x.size(1), -1)  
        return x

class VQVAE_PITCH(nn.Module):
    def __init__(
        self,
        in_channel=80,
        channel=512,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
        embed_pre = 256
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
            Quantize(in_channel//2**(i+1), n_embed)]
        self.quantize = nn.ModuleList(quantize_blocks)
        
        
        self.dec = Decoder_PITCH(
            in_channel ,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=8,
        )
    def forward(self, input):
        enc_b, sp_embed, std_block, pitch_block, diff = self.encode(input_pitch, input)
        #print (enc_speaker.size(), enc_b.size(), enc_b_content.size())
        dec_1= self.decode(enc_b, sp_embed, std_block, pitch_block)


        #enc_b2, sp_embed2, std_block2, rhy_block2, diff2 = self.encode(input, input_rhy)
        #print (enc_speaker.size(), enc_b.size(), enc_b_content.size())
        #dec_2= self.decode(enc_b2, sp_embed2, std_block2, rhy_block2)
        idx = torch.randperm(enc_b[0].size(0))
        
        sp_shuffle = []
        std_shuffle = []
        pitch_shuffle = []
        
        for sm in (sp_embed):
            sp_shuffle += [sm[idx]]
        for std in std_block:
            std_shuffle += [std[idx]]
        for pit in pitch_block:
            pitch_shuffle += [pit[idx]]

        dec_2 = self.decode(enc_b, sp_shuffle, std_shuffle, pitch_shuffle)
        
        
        return dec_1, dec_2, enc_b, sp_embed, diff, idx
    def resample(self, input):
        #input = F.interpolate(input, scale_factor=0.5, mode='linear')
        #input = F.interpolate(input, scale_factor=2, mode='nearest')
        #input = F.interpolate(input, scale_factor=1/4, mode='linear')
        #input = F.interpolate(input, scale_factor=4, mode='nearest')
        #input = F.interpolate(input, scale_factor=1/8, mode='linear')
        #input = F.interpolate(input, scale_factor=8, mode='nearest')

        return input
    def encode(self, input_pitch, input):
        x = input
        x_pitch = input_pitch
        sp_embedding_block = []
        q_after_block = []
        rhythm_block = []
        std_block = []
        diff_total = 0


        for i, (enc_block, quant) in enumerate(zip(self.enc, self.quantize)):
            x = enc_block(x)   
            #x_pitch = enc_block(x_pitch)

            pitch = torch.mean(x, dim = 2, keepdim = True)
            
            x_ = x - torch.mean(x, dim = 2, keepdim = True)
            std_ = torch.norm(x_, dim= 2, keepdim = True) + 1e-4
            std_block += [std_]
            x_ = x_ / std_
            
            x_ = x_ / torch.norm(x_, dim = 1, keepdim = True)
            q_after, diff = quant(x_.permute(0,2,1))
            q_after = q_after.permute(0,2,1)
            
            sp_embed = torch.mean(x - q_after, 2, True)
            sp_embed = sp_embed / (torch.norm(sp_embed, dim = 1, keepdim=True)+1e-4) /3

            q_after = self.resample(q_after)
            q_after = q_after - torch.mean(q_after, dim= 1 , keepdim = True)
            sp_embedding_block += [sp_embed]
            q_after_block += [q_after]
            pitch_block += [pitch]
            diff_total += diff
        
        return q_after_block, sp_embedding_block, std_block, pitch_block, diff_total 

    def decode(self, quant_b, sp, std, rhy):
        
        dec_1 = self.dec(quant_b, sp, std, rhy)
        
        return dec_1
