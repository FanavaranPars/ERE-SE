import numpy as np
import librosa
import random
import pickle
import os
import torch

from torch.utils.data import Dataset, DataLoader
from speechbrain.processing.signal_processing import reverberate
from torch.nn.utils.rnn import pad_sequence

def creat_data_pathes(
                    base_path,
                    groups,
                    g_scale,
                    filenames = "train_files_path.txt"
                    ):
    total_paths = []
    i = 0
    for group_name in groups:
        with open( os.path.join(base_path,group_name,filenames ), 'rb') as fp:
            pathes = pickle.load(fp)
        pathes = pathes * g_scale[i]
        for path in pathes:
            total_paths.append(os.path.join(group_name,path))
        
        i+=1
    
    random.seed(12)
    random.shuffle(total_paths)
    random.shuffle(total_paths)
    return total_paths 


class Enhancement_DATASET(Dataset):
    def __init__(self,
                 base_clean_path,
                 base_noise_path,
                 base_rever_path,
                 clean_paths,
                 noise_paths,
                 reverb_paths,
                 sampling_rate = 16000,
                 max_length = 10 * 16000,
                 max_noise_n = 2, #max = 2
                 t_reverb = 0.5,
                 min_snr = -10
                ):
        
        self.base_clean_path = base_clean_path
        self.base_noise_path = base_noise_path
        self.base_rever_path = base_rever_path
        self.clean_paths = clean_paths
        self.noise_paths = noise_paths
        self.reverb_paths = reverb_paths
        self.len_clean = len(clean_paths)
        self.len_noise = len(noise_paths)
        self.len_reverb = len(reverb_paths)
        
        self.sampling_rate = sampling_rate
        self.max_length = max_length
        self.max_noise_n = max_noise_n
        self.t_reverb = t_reverb
        
        self.len_snr = len(range(min_snr,31,2))
        self.SNR_amount = range(min_snr,31,2)
        
        print("Dataset is ready.")
    
    def load_sample(self, path):
        
        waveform, _ = librosa.load(path, sr=self.sampling_rate,  dtype='float32')
        return waveform
    
    def create_reverb(self, sig, reverb_filename): 
        reverb_ = torch.from_numpy(self.load_sample(reverb_filename)).float()
        reverb_sig = reverberate(sig.unsqueeze(dim = 0), reverb_, rescale_amp= 'peak')

        return reverb_sig.squeeze()
    
    def crop_noise(self, noise, len_x):
        len_n = len(noise)
        extra = len_n - len_x
        if extra > 0:
            first_ind = random.randint(0,extra - 1)
            noise = noise[first_ind:first_ind+len_x]
        
        return noise
    
    def crop_audio(self, x):
        len_x = len(x)
        extra = len_x - self.max_length
        if extra > 0:
            first_ind = random.randint(0,extra - 1)
            x = x[first_ind:first_ind+self.max_length]
            len_x = self.max_length
        
        return x, len_x
    
    def prepare_noise(self, path, len_x):
        noise = self.load_sample(path)
        len_n = len(noise)
        if len_n < len_x:
            repeat = len_x // len_n + 1 
            noise = [noise for _ in range(repeat)]
            noise = np.concatenate(noise, axis=0)

        noise = self.crop_noise(noise, len_x)
        return noise
    
    def creat_noisy_data(self, x_clean, noise, SNR):
        sp_ener = torch.sum(x_clean**2)
        noi_ener = torch.sum(noise**2) + 1e-8
        a = (sp_ener/noi_ener)**0.5 * 10**(-SNR/20)
        x_noisy = x_clean + a * noise
        return x_noisy
    
        
    def __len__(self):
        return len(self.clean_paths)
        # return 100
        

    def __getitem__(self, index):
        # load to tensors and normalization
        x_clean = self.load_sample(
                                os.path.join(self.base_clean_path,
                                self.clean_paths[index])
                                )
        c = np.sqrt(len(x_clean) / np.sum(x_clean ** 2.0))
        x_clean, len_x = self.crop_audio(x_clean)
        # x_clean = x_clean * np.random.uniform(0.7,1,1)
        noise = self.prepare_noise(
                                    os.path.join(self.base_noise_path,
                                    self.noise_paths[random.sample(range(self.len_noise),1)[0]]),
                                    len_x
                                    )
        
        x_clean_add = torch.from_numpy(x_clean).float()
        x_clean = torch.from_numpy(x_clean).float()
        noise = torch.from_numpy(noise).float()
        
        is_reverb = torch.rand(1) < self.t_reverb
        
        if is_reverb:
            x_clean_add = self.create_reverb(
                                        x_clean_add,
                                        os.path.join(self.base_rever_path,
                                        self.reverb_paths[random.sample(range(self.len_reverb),1)[0]])
                                            )
            noise = self.create_reverb(
                                        noise,
                                        os.path.join(self.base_rever_path,
                                        self.reverb_paths[random.sample(range(self.len_reverb),1)[0]])
                                        )
        
        n_o_n = random.randint(1,self.max_noise_n)
        if n_o_n == 2:
            noise_2 = self.prepare_noise(
                                        os.path.join(self.base_noise_path,
                                        self.noise_paths[random.sample(range(self.len_noise),1)[0]]),
                                        len_x,
                                        )
            
            noise_2 = torch.from_numpy(noise_2).float()
            if is_reverb:
                noise_2 = self.create_reverb(
                                            noise,
                                            os.path.join(self.base_rever_path,
                                            self.reverb_paths[random.sample(range(self.len_reverb),1)[0]]),
                                            )
            noise = noise + noise_2
            
        
        snr = self.SNR_amount[random.sample(range(self.len_snr),1)[0]]
        x_noisy = self.creat_noisy_data(x_clean_add, noise, snr)

        return x_noisy*c, x_clean*c, x_clean_add, is_reverb, n_o_n, snr

class Enhancement_evaluation_DATASET(Dataset):
    def __init__(self,
                 base_clean_path,
                 base_eval_pth,
                 eval_filenames,
                 information_dict,
                 sampling_rate = 16000,
                ):
        
        self.base_clean_path = base_clean_path
        self.base_eval_pth = base_eval_pth
        self.eval_filenames = eval_filenames
        self.information_dict = information_dict
        
        self.sampling_rate = sampling_rate
    
    def load_sample(self, path):
        waveform, _ = librosa.load(path, sr=self.sampling_rate,  dtype='float32')
        return waveform
        
    def __len__(self):
        return len(self.eval_filenames)
        # return 50
        

    def __getitem__(self, index):
        # load to tensors and normalization
        num, data_name, snr, f_n_n,\
              s_n_n, c_r_n, f_r_n_n, s_r_n_n = self.eval_filenames[index][:-4].split("_")
        
        x_noisy = self.load_sample(
                                    os.path.join(self.base_eval_pth,
                                    data_name,
                                    self.eval_filenames[index]),
                                    )
        
        inform = self.information_dict[int(num)]
        clean_path = inform["clean_path"]
        x_clean = self.load_sample(os.path.join(self.base_clean_path,
                                                clean_path))
        
        x_clean = torch.from_numpy(x_clean).float()
        x_noisy = torch.from_numpy(x_noisy).float()
        
        return x_noisy, x_clean, inform, "", "", ""
        
        
        
def collate_fn(batch):
    inputs, targets, length_ratio = [], [], []
    for noisy_input, clean_target, _, _, _, _ in batch:
        inputs.append(noisy_input)
        targets.append(clean_target)
        length_ratio.append(len(noisy_input))

    inputs = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0.0)

    length_ratio = torch.tensor(length_ratio, dtype=torch.long) / inputs.shape[1]

    return inputs, targets, length_ratio
    
def evaluation_data_loader(
                        base_clean_path,
                        base_eval_pth,
                        eval_filenames,
                        information_dict,
                        sampling_rate,
                        batch_size, 
                        num_workers, 
                        pin_memory,
                        ):
        
    dataset = Enhancement_evaluation_DATASET(
                                            base_clean_path,
                                            base_eval_pth,
                                            eval_filenames,
                                            information_dict,
                                            sampling_rate,
                                            )
    
    loader = DataLoader(
                        dataset,
                        batch_size = batch_size,
                        shuffle = False,
                        drop_last = True,
                        collate_fn = collate_fn,
                        num_workers = num_workers,
                        pin_memory = pin_memory
                        )
    
    return loader 
# for reading and preparing dataset
def audio_data_loader(
                    base_clean_path,
                    base_noise_path,
                    base_rever_path,
                    clean_paths,
                    noise_paths,
                    reverb_paths,
                    sampling_rate,
                    max_length,
                    max_noise_n,
                    t_reverb,
                    min_snr,
                    batch_size, 
                    num_workers, 
                    pin_memory,
                    training
                      ):
        
    dataset = Enhancement_DATASET(base_clean_path,
                                base_noise_path,
                                base_rever_path,
                                clean_paths,
                                noise_paths,
                                reverb_paths,
                                sampling_rate,
                                max_length,
                                max_noise_n,
                                t_reverb,
                                min_snr
                                )
    
    loader = DataLoader(dataset,
                        batch_size = batch_size,
                        shuffle = training,
                        drop_last = True,
                        collate_fn = collate_fn,
                        num_workers = num_workers,
                        pin_memory = pin_memory
                        )
    
    return loader