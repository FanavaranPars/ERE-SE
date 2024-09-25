
import numpy as np

import torch
import torch.nn as nn

from speechbrain.dataio.preprocess import AudioNormalizer



class ISTFT(nn.Module):
    def __init__(self,
                 n_fft=400,
                 hop_length=160,
                 window="hamming_window",
                 normalized=False):
        super(ISTFT, self).__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalized = normalized
        self.window = getattr(torch, window)(n_fft)
        
        
    def forward(self, x):
        x = torch.view_as_complex(x)

        x_istft = torch.istft(x,
                              n_fft=self.n_fft,
                              hop_length=self.hop_length,
                              window=self.window.to(x.device),
                              normalized=self.normalized)
        
        return x_istft    



class STFT(nn.Module):
    def __init__(self,
                 n_fft=400,
                 hop_length=160,
                 window="hamming_window",
                 normalized=False,
                 pad_mode="constant",
                 return_complex=False):
        super(STFT, self).__init__()
        
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalized = normalized
        self.return_complex = return_complex
        self.pad_mode = pad_mode
        self.window = getattr(torch, window)(n_fft)
        
        
    def forward(self, x):
        x_stft = torch.stft(input=x,
                            n_fft=self.n_fft, 
                            hop_length=self.hop_length,
                            window=self.window.to(x.device),
                            normalized=self.normalized,
                            pad_mode=self.pad_mode,
                            return_complex=self.return_complex)
        
        return x_stft 
        
        
        
def load_asr_encoder(path, device):
    network = torch.load(path)
    
    encoder = network["encoder"]
    encoder.compute_features.compute_STFT = torch.nn.Identity()

    if device == "cuda":
        encoder = encoder.cuda()
        encoder.normalize.glob_mean = encoder.normalize.glob_mean.cuda()
        encoder.normalize.glob_std = encoder.normalize.glob_std.cuda()
        
    return encoder
    

    

class EnhancemetPipline(nn.Module):
    def __init__(self,
                 enhacement,
                 asr_encoder,
                 stft_layer,
                 ):
        super(EnhancemetPipline, self).__init__()
        self.enhacement = enhacement
        self.asr_encoder = asr_encoder
        self.stft_layer = stft_layer
        self.normalizer = AudioNormalizer(16000)


    def forward(self, noisy_input, clean_target, length_ratio):

        clean_target= self.normalizer(clean_target.unsqueeze(dim=1),16000)
        noisy_input= self.normalizer(noisy_input.unsqueeze(dim=1),16000)

        noisy_stft = self.stft_layer(noisy_input).permute([0,3,1,2])
        clean_stft = self.stft_layer(clean_target)

        esti_list = self.enhacement(noisy_stft)
        enhancement_output = esti_list[-1].permute([0,3,2,1])

        noisy_embed = self.asr_encoder(enhancement_output, length_ratio)
        target_embed = self.asr_encoder(clean_stft, length_ratio)

        return noisy_embed, target_embed, clean_stft.permute(0,3,2,1), esti_list


class EnhancemetSolo(nn.Module):
    def __init__(self,
                 enhacement,
                 stft_layer,
                 ):
        super(EnhancemetSolo, self).__init__()
        self.enhacement = enhacement
        self.stft_layer = stft_layer
        self.normalizer = AudioNormalizer(16000)


    def forward(self, noisy_input, clean_target, length_ratio):

        clean_target= self.normalizer(clean_target.unsqueeze(dim=1),16000)
        noisy_input= self.normalizer(noisy_input.unsqueeze(dim=1),16000)
        
        noisy_stft = self.stft_layer(noisy_input)
        target_stft = self.stft_layer(clean_target)

        noisy_stft, target_stft = noisy_stft.permute(0,3,2,1), target_stft.permute(0,3,1,2)

        noisy_mag, noisy_phase = torch.norm(noisy_stft, dim=1) ** 0.5, torch.atan2(noisy_stft[:, -1, ...], noisy_stft[:, 0, ...])
        target_mag, target_phase = torch.norm(target_stft, dim=1) ** 0.5, torch.atan2(target_stft[:, -1, ...], target_stft[:, 0, ...])
        noisy_stft = torch.stack((noisy_mag * torch.cos(noisy_phase), noisy_mag * torch.sin(noisy_phase)), dim=1)
        target_stft = torch.stack((target_mag * torch.cos(target_phase), target_mag * torch.sin(target_phase)), dim=1)

        esti_list = self.enhacement(noisy_stft)

        return target_stft, esti_list
    

class Wave_Enhancement(nn.Module):
    def __init__(self,
                 enhacement,
                 stft_layer,
                 istft_layer
                 ):
        super(Wave_Enhancement, self).__init__()
        self.enhacement = enhacement
        self.stft_layer = stft_layer
        self.istft_layer = istft_layer
        self.normalizer = AudioNormalizer(16000)


    def forward(self, noisy_input):
        
        #noisy_input= self.normalizer(noisy_input.unsqueeze(dim=1),16000)
        noisy_stft = self.stft_layer(noisy_input)
        noisy_stft = noisy_stft.permute(0,3,2,1)
        
        noisy_mag, noisy_phase = torch.norm(noisy_stft, dim=1) ** 0.5, torch.atan2(noisy_stft[:, -1, ...], noisy_stft[:, 0, ...])
        noisy_stft = torch.stack((noisy_mag * torch.cos(noisy_phase), noisy_mag * torch.sin(noisy_phase)), dim=1)
        
        rec_stft = self.enhacement(noisy_stft)[-1].permute([0,2,3,1])

        est_mag, est_phase = torch.norm(rec_stft, dim=-1)**2.0,torch.atan2(rec_stft[..., -1], rec_stft[...,0])
        est_stft = torch.stack((est_mag*torch.cos(est_phase),est_mag*torch.sin(est_phase)), dim=-1)
        
        noisy_rec = self.istft_layer(est_stft)
        return noisy_rec