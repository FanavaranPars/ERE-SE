import torch
import torch.nn.functional as F
import argparse
import librosa
import os
import glob
import numpy as np
from models.istft import ISTFT_DB
from models.dbaiat import dual_aia_trans_merge_crm
from models.utils import create_enhancement_model_and_configs
from models.utils import ISTFT, STFT, EnhancementW2W
import soundfile as sf
import librosa
import warnings
warnings.filterwarnings('ignore')

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
def enhance_dbaiat(args):
    model_name = args.model
    device = args.device
    mix_file_path = args.mix_file_path
    esti_file_path = args.esti_file_path
    model_root_path = args.model_path
    checkpoint_path = model_root_path + model_name + '.pt'
    model = dual_aia_trans_merge_crm()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    with torch.no_grad():
        cnt = 0        
        file_list = os.listdir(mix_file_path)
        istft = ISTFT_DB(filter_length=320, hop_length=160, window='hann')
        for file_id in file_list:
            feat_wav, _ = librosa.load(os.path.join(mix_file_path, file_id), sr=args.fs)
            feat_wav = feat_wav[:len(feat_wav)//1]
            c = np.sqrt(len(feat_wav) / np.sum((feat_wav ** 2.0)))
            feat_wav = feat_wav * c
            wav_len = len(feat_wav)
            frame_num = int(np.ceil((wav_len - 320 + 320) / 160 + 1))
            fake_wav_len = (frame_num - 1) * 160 + 320 - 320
            left_sample = fake_wav_len - wav_len
            feat_wav = torch.FloatTensor(np.concatenate((feat_wav, np.zeros([left_sample])), axis=0))
            feat_x = torch.stft(feat_wav.unsqueeze(dim=0), n_fft=320, hop_length=160, win_length=320,
                                window=torch.hann_window(320), return_complex=False).permute(0, 3, 2, 1)
            noisy_phase = torch.atan2(feat_x[:, -1, :, :], feat_x[:, 0, :, :])
            feat_x_mag = (torch.norm(feat_x, dim=1)) ** 0.5
            feat_x = torch.stack((feat_x_mag * torch.cos(noisy_phase), feat_x_mag * torch.sin(noisy_phase)), dim=1)
            esti_x = model(feat_x.to(device))
            esti_mag, esti_phase = torch.norm(esti_x, dim=1), torch.atan2(esti_x[:, -1, :, :],
                                                                             esti_x[:, 0, :, :])
            esti_mag = esti_mag ** 2
            esti_com = torch.stack((esti_mag * torch.cos(esti_phase), esti_mag * torch.sin(esti_phase)), dim=1)
            esti_com = esti_com.cpu()
            esti_utt = istft(esti_com).squeeze().numpy()
            esti_utt = esti_utt[:wav_len]
            esti_utt = esti_utt / c
            os.makedirs(os.path.join(esti_file_path, model_name), exist_ok=True)
            sf.write(os.path.join(esti_file_path, model_name, file_id), esti_utt, args.fs)
            print(f' The {cnt+1} utterance was enhanced by {model_name}!')
            cnt += 1

def enhance_gagnet(args):
    path_list = []
    model_name = args.model
    device = args.device
    mix_file_path = args.mix_file_path
    esti_file_path = args.esti_file_path
    model_root_path = args.model_path
    checkpoint_path = model_root_path + model_name + '.pt'
    for path in glob.glob(os.path.join(mix_file_path+"/*")):
        path_list.append(path)

    stft_layer = STFT().to(device)
    istft_layer = ISTFT().to(device)


    enhacement, model_configs = create_enhancement_model_and_configs(model_name = model_name,
                                                                    DEVICE = device)

    enhacement.load_state_dict(torch.load(checkpoint_path))
    enh_model = EnhancementW2W(enhacement, stft_layer, istft_layer)


    sr=16000
    cnt = 0
    for path_audio in path_list:
        
        sig, sr = librosa.load(path_audio,sr=sr, dtype='float32')

        enh_model.eval()

        torch.cuda.empty_cache()
        with torch.no_grad():
            enhanced_sig = enh_model(torch.from_numpy(sig).unsqueeze(dim = 0).to(device)).cpu().numpy()
        
        os.makedirs(os.path.join(esti_file_path, model_name), exist_ok=True)
        sf.write(os.path.join(esti_file_path, model_name ,path_audio.split("/")[-1]) , 
                        enhanced_sig[0], sr)
        print(f' The {cnt+1} utterance was enhanced by {model_name}!')
        cnt += 1
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Enhancing Audio')

    parser.add_argument('-model', type=str, default='dbaiat',
                        help="You can choose your enhancement model from ['dbaiat','gagnet']")
    
    parser.add_argument('-device', type=str, default='cuda',
                        help="You can choose your device from ['cuda', 'cpu']")
    
    parser.add_argument('--mix_file_path', type=str, default='./Noisy_Samples/')
    parser.add_argument('--esti_file_path', type=str, default='./Enhanced_Samples/')    #  -5  -2  0  2  5
    parser.add_argument('--fs', type=int, default=16000,
                        help='The sampling rate of speech')
    parser.add_argument('--model_path', type=str, default='./checkpoints/',
                        help='The place to save best model')
    args = parser.parse_args()
    
    if args.model == 'dbaiat':
        enhance_dbaiat(args=args)
    if args.model == 'gagnet':
        enhance_gagnet(args=args)    