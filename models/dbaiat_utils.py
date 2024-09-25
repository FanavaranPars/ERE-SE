import torch
import torch.nn as nn
from models.dbaiat import dual_aia_trans_merge_crm

def create_enhancement_model_and_configs(model_configs, DEVICE = "cuda"):
    """Create the enhancement model and its configs

    Arguments
    ---------
    model_name : str
        Name of enhancement model ("gagnet"").
    DEVICE : str
        GPU ("cuda") or CPU ("cpu").

    Returns
    -------
    enhacement : class
        The enhancement model.
    model_configs : dict
        The enhancement model configs.

    """

    if model_configs["name"] == "dbaiat":

        enhacement = dual_aia_trans_merge_crm().to(DEVICE)
    
    else:
        raise ValueError("the name of the model is not supported!!")

    

    return enhacement, model_configs

class ISTFT(nn.Module):
    """Inverse short-time Fourier transform (ISTFT).

    Arguments
    ---------
        input : float (Tensor)
            The input tensor. Expected to be in the format of :func:`~torch.stft`, output.
            That is a complex tensor of shape `(B?, N, T)` where 
            - `B?` is an optional batch dimension
            - `N` is the number of frequency samples, `(n_fft // 2) + 1`
                for onesided input, or otherwise `n_fft`.
            - `T` is the number of frames, `1 + length // hop_length` for centered stft,
                or `1 + (length - n_fft) // hop_length` otherwise.
        n_fft : int
            Size of Fourier transform.
        hop_length : int
            The distance between neighboring sliding window.
        window : str
           The optional window function.
        normalized : bool
            controls whether to return the normalized ISTFT results

    Returns
    -------
        x_istft : float (Tensor)
            ISTFT of the input.

    """
    def __init__(self,
                 n_fft=320,
                 hop_length=160,
                 window="hann_window",
                 normalized=False):
        super(ISTFT, self).__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalized = normalized
        self.window = getattr(torch, window)(n_fft)
        
    def forward(self, x):
        """This method should implement forwarding operation in the ISTFT.

        Arguments
        ---------
        x : float (Tensor)
            The input of ISTFT.

        Returns
        -------
        x_istft : float (Tensor)
            The output of ISTFT.
        """
        x = torch.view_as_complex(x)

        x_istft = torch.istft(x,
                              n_fft=self.n_fft,
                              hop_length=self.hop_length,
                              window=self.window.to(x.device),
                              normalized=self.normalized)
        
        return x_istft    



class STFT(nn.Module):
    """Short-time Fourier transform (STFT).

    Arguments
    ---------
        input : float (Tensor)
            The input tensor of shape `(B, L)` where `B` is an optional.
        n_fft : int
            Size of Fourier transform.
        hop_length : int
            The distance between neighboring sliding window.
        window : str
           The optional window function.
        normalized : bool
            Controls whether to return the normalized STFT results.
        pad_mode : str
            controls the padding method used.
        return_complex : bool
            Whether to return a complex tensor, or a real tensor with 
            an extra last dimension for the real and imaginary components.

    Returns
    -------
        x_istft : float (Tensor)
            STFT of the input.

    """
    def __init__(self,
                 n_fft=320,
                 win_length=320,
                 hop_length=160,
                 window="hann_window",
                 return_complex=False):
        super(STFT, self).__init__()
        
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.return_complex = return_complex
        self.window = getattr(torch, window)(n_fft)
        
        
    def forward(self, x):
        """This method should implement forwarding operation in the STFT.

        Arguments
        ---------
        x : float (Tensor)
            The input of STFT.

        Returns
        -------
        x_stft : float (Tensor)
            The output of STFT.
        """
        x_stft = torch.stft(input=x,
                            n_fft=self.n_fft,
                            win_length=self.win_length, 
                            hop_length=self.hop_length,
                            window=self.window.to(x.device),
                            return_complex=self.return_complex)
        
        return x_stft 
        
    

def preprocessing_for_dbaiat_train(noisy_stft, clean_stft, feat_type='sqrt'):
    """Pre-processing of GAGNet models for input and target of them.

    Arguments
    ---------
        noisy_stft : float (Tensor)
            STFT of the input of GAGNet models.
        target_stft : float (Tensor)
            STFT of the target of GAGNet models.

    Returns
    -------
        c_noisy_stft : float (Tensor)
            Changed STFT of the input of GAGNet models.
        c_target_stft : float (Tensor)
            Changed STFT of the target of GAGNet models.

    """
    noisy_stft = noisy_stft.permute(0,3,2,1)
    clean_stft = clean_stft.permute(0,3,2,1)
    
    noisy_phase = torch.atan2(noisy_stft[:,-1,:,:], noisy_stft[:,0,:,:])
    clean_phase = torch.atan2(clean_stft[:,-1,:,:], clean_stft[:,0,:,:])


    # three approachs for feature compression:
    if feat_type == 'normal':
        noisy_stft, clean_stft = torch.norm(noisy_stft, dim=1), torch.norm(clean_stft, dim=1)
    elif feat_type == 'sqrt':
        noisy_stft, clean_stft = (torch.norm(noisy_stft, dim=1)) ** 0.5, (
            torch.norm(clean_stft, dim=1)) ** 0.5
    elif feat_type == 'cubic':
        noisy_stft, clean_stft = (torch.norm(noisy_stft, dim=1)) ** 0.3, (
            torch.norm(clean_stft, dim=1)) ** 0.3
    elif feat_type == 'log_1x':
        noisy_stft, clean_stft = torch.log(torch.norm(noisy_stft, dim=1) + 1), \
                                    torch.log(torch.norm(clean_stft, dim=1) + 1)

    noisy_stft = torch.stack((noisy_stft*torch.cos(noisy_phase), noisy_stft*torch.sin(noisy_phase)), dim=1)
    clean_stft = torch.stack((clean_stft*torch.cos(clean_phase), clean_stft*torch.sin(clean_phase)), dim=1)


    return noisy_stft, clean_stft

def preprocessing_for_dbaiat(noisy_stft):
    """Pre-processing of GAGNet models for input of them.

    Arguments
    ---------
        noisy_stft : float (Tensor)
            STFT of the input of GAGNet models.

    Returns
    -------
        c_noisy_stft : float (Tensor)
            Changed STFT of the input of GAGNet models.

    """
    noisy_stft = noisy_stft.permute(0,3,2,1)
    noisy_mag = torch.norm(noisy_stft, dim=1) ** 0.5
    noisy_phase = torch.atan2(noisy_stft[:, -1, ...], noisy_stft[:, 0, ...])
    
    c_noisy_stft = torch.stack((noisy_mag * torch.cos(noisy_phase),
                               noisy_mag * torch.sin(noisy_phase)), dim=1)

    return c_noisy_stft

def postprocessing_for_dbaiat(enhancement_output):
    """Post-processing of GAGNet models for output of them.

    Arguments
    ---------
        enhancement_output : float (Tensor)
            Output of GAGNet models.

    Returns
    -------
        c_enhancement_output : float (Tensor)
            Changed output of GAGNet models.

    """

    est_mag = torch.norm(enhancement_output, dim=-1)**2.0
    est_phase = torch.atan2(enhancement_output[..., -1], enhancement_output[...,0])
    c_enhancement_output = torch.stack((est_mag*torch.cos(est_phase),
                                      est_mag*torch.sin(est_phase)), dim=-1)

    return c_enhancement_output


class Enhancemet_dbaiat_train(nn.Module):
    """Enhancement training without encoder Hamrah.

    Arguments
    ---------
        enhancement : class
            GAGNet models.
        stft_layer : class
            STFT module.

    Returns
    -------
        esti_list : float (Tensor)
            Output of each defined layer of GAGNet for computing enhancement loss.
        target_stft : float (Tensor)
            Pre-processed STFT of the clean input as a target for computing enhancement loss.

    """
    def __init__(self,
                 enhancement,
                 stft_layer,
                 feat_type='sqrt',
                 n_fft=320,
                 win_size=320,
                 win_shift=160
                 ):
        super(Enhancemet_dbaiat_train, self).__init__()
        self.enhancement = enhancement
        self.stft_layer = stft_layer
        self.feat_type = feat_type

        self.n_fft = n_fft
        self.win_size = win_size
        self.win_shift = win_shift

    def forward(self, noisy_input, clean_target, length_ratio):
        """This method should implement forwarding operation in the EnhancemetSolo_train.

        Arguments
        ---------
        noisy_input : float (Tensor)
            The noisy input of EnhancemetSolo_train.
        clean_target : float (Tensor)
            The clean input of EnhancemetSolo_train.

        Returns
        -------
        esti_list : float (Tensor)
            Output of each defined layer of GAGNet for computing enhancement loss.
        target_stft : float (Tensor)
            Pre-processed STFT of the clean input as a target for computing enhancement loss.

        """
        wave_length = length_ratio * noisy_input.shape[1]
        wave_length = wave_length.int()
        frame_num = (wave_length - self.win_size + self.n_fft) // self.win_shift + 1
        # print(frame_num)
        # print(noisy_input.shape)

        noisy_stft = self.stft_layer(noisy_input)
        clean_stft = self.stft_layer(clean_target)
        
        noisy_stft, target_stft = preprocessing_for_dbaiat_train(noisy_stft, clean_stft, self.feat_type)
        esti_list = self.enhancement(noisy_stft)

        return esti_list, target_stft, frame_num
    

# class EnhancementW2W(nn.Module):
#     """End-to-End Enhancement module.

#     Arguments
#     ---------
#         enhancement : class
#             GAGNet models.
#         stft_layer : class
#             STFT module.
#         istft_layer : class
#             ISTFT module.

#     Returns
#     -------
#         noisy_rec : float (Tensor)
#             Enhanced of the input audio

#     """
#     def __init__(self,
#                  enhancement,
#                  stft_layer,
#                  istft_layer
#                  ):
#         super(EnhancementW2W, self).__init__()
#         self.enhancement = enhancement
#         self.stft_layer = stft_layer
#         self.istft_layer = istft_layer
#         self.normalizer = AudioNormalizer(16000)


#     def forward(self, noisy_input):
#         """This method should implement forwarding operation in the EnhancementW2W.

#         Arguments
#         ---------
#         noisy_input : float (Tensor)
#             The noisy input of EnhancementW2W.

#         Returns
#         -------
#         noisy_rec : float (Tensor)
#             Enhanced of the input audio

#         """
        
#         noisy_input= self.normalizer(noisy_input.unsqueeze(dim=1),16000)
#         noisy_stft = self.stft_layer(noisy_input)
#         noisy_stft = preprocessing_for_GAGNet(noisy_stft)

#         rec_stft = self.enhancement(noisy_stft)[-1].permute([0,2,3,1])

#         est_stft = postprocessing_for_GAGNet(rec_stft)
#         noisy_rec = self.istft_layer(est_stft)
        
#         return noisy_rec

