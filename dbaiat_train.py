#import argparse
import os
import torch
from models.dbaiat_utils import create_enhancement_model_and_configs
from recipes.utils import load_model_config
from tools.train_dbaiat_utils import run, dbaiat_loss_fn
from dataio.dbaiat_dataio import (creat_data_pathes,audio_data_loader)
from models.dbaiat_utils import STFT, Enhancemet_dbaiat_train
import warnings
warnings.filterwarnings('ignore')



# use cuda if cuda available 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))

# construct the argument parser and parse the arguments

model_name = "dbaiat"#args["name_of_enhancement_model"]
data_cfg = "./recipes/dataio/dataio_dbaiat.json" #args["data_config_path"]
train_cfg = "./recipes/training/training_dbaiat.json" #args["train_config_path"]
model_cfg = "./recipes/models/dbaiat_finetune.json"
# DATA PATHS
base_data_pth = "/home/dllabsharif/Documents/DATASETS"

base_clean_pth = os.path.join(base_data_pth,"clean_dataset")
base_noise_pth = os.path.join(base_data_pth,"Noise_dataset")
base_rever_pth = os.path.join(base_data_pth,"Noise_dataset/RIRS_NOISES")

clean_groups = ["CommonVoice", "60 Giga", "TIMIT"]
clean_scalse = [1,1,10]

## Fine-Tune Just on CommonVoice
clean_groups = ["CommonVoice","TIMIT"]
clean_scalse = [1,4]

noise_groups = ["QUT", "pointsource_noises", "musan_noise", "Audioset"]
noise_scalse = [1,5,10,2]
noise_scalse_val =  [3,20,50,1] 

rirs_groups = ["simulated_rirs", "real_iris"]
rirs_scalse = [1,50]

# LOAD DATA PATHS
clean_train_paths = creat_data_pathes(base_clean_pth,
                                      clean_groups,
                                      clean_scalse,
                                      filenames = "train_files_path.txt")

noise_train_paths = creat_data_pathes(base_noise_pth,
                                      noise_groups,
                                      noise_scalse,
                                      filenames = "train_files_path.txt")

rirs_train_paths = creat_data_pathes(base_rever_pth,
                                      rirs_groups,
                                      rirs_scalse,
                                      filenames = "train_files_path.txt")

print(f"number of clean train : {len(clean_train_paths)}\n \
        number of noise of train : {len(noise_train_paths)} \n \
        number of reverb of train :       {len(rirs_train_paths)}")


clean_valid_paths = creat_data_pathes(base_clean_pth,
                                      clean_groups,
                                        clean_scalse,
                                        filenames = "validation_files_path.txt")

noise_valid_paths = creat_data_pathes(base_noise_pth,
                                      noise_groups,
                                      noise_scalse_val,
                                      filenames = "validation_files_path.txt")

rirs_valid_paths = creat_data_pathes(base_rever_pth,
                                      rirs_groups,
                                      rirs_scalse,
                                      filenames = "validation_files_path.txt")

print(f"number of clean val : {len(clean_valid_paths)}\n \
        number of noise of val : {len(noise_valid_paths)} \n \
        number of reverb of val :       {len(rirs_valid_paths)}")

data_configs = load_model_config(data_cfg)

train_dataset = audio_data_loader(base_clean_pth,
                                base_noise_pth,
                                base_rever_pth,
                                clean_train_paths,
                                noise_train_paths,
                                rirs_train_paths,
                                data_configs["train"]["SAMPLE_RATE"],
                                data_configs["train"]["MAX_LENGTH"],
                                data_configs["train"]["MAX_NOISE_N"], #max = 2
                                data_configs["train"]["T_REVERB"], # no reverb = -1
                                data_configs["train"]["MIN_SNR"],
                                data_configs["train"]["BATCH_SIZE"], 
                                data_configs["train"]["NUM_WORKER"], 
                                data_configs["train"]["PIN_MEMORY"],
                                data_configs["train"]["TRAINING"]
                                )

valid_dataset = audio_data_loader(base_clean_pth,
                                base_noise_pth,
                                base_rever_pth,
                                clean_valid_paths,
                                noise_valid_paths,
                                rirs_valid_paths,
                                data_configs["test"]["SAMPLE_RATE"],
                                data_configs["test"]["MAX_LENGTH"],
                                data_configs["test"]["MAX_NOISE_N"], #max = 2
                                data_configs["test"]["T_REVERB"], # no reverb = -1
                                data_configs["test"]["MIN_SNR"],
                                data_configs["test"]["BATCH_SIZE"], 
                                data_configs["test"]["NUM_WORKER"], 
                                data_configs["test"]["PIN_MEMORY"],
                                data_configs["test"]["TRAINING"]
                                )

training_configs = load_model_config(train_cfg)


stft_layer = STFT().to(DEVICE)

model_configs = load_model_config(model_cfg)

enhacement, model_configs = create_enhancement_model_and_configs(
                                                        model_configs = model_configs,
                                                        DEVICE = DEVICE
                                                        )

loss_fn = dbaiat_loss_fn

if training_configs["IS_FINETUNE"]:
    enhacement.load_state_dict(torch.load(model_configs["param_best_path"]))

enh_pipline = Enhancemet_dbaiat_train(enhacement, stft_layer)


optimizer = torch.optim.Adam(enh_pipline.parameters(),
                              lr=training_configs["LEARNING_RATE"])

train_loss, vall_loss, best_valid_loss = run(
                                            enh_pipline,
                                            train_dataset,
                                            valid_dataset,
                                            optimizer,
                                            loss_fn,
                                            save_model_path= model_configs["param_save_path"],
                                            chkp_path = model_configs["loss_save_path"],
                                            step_show= training_configs["STEP_SHOW"],
                                            n_epoch= training_configs["NUM_EPOCH"],
                                            grad_acc_step= training_configs["GRAD_STEP"],
                                            is_finetune = training_configs["IS_FINETUNE"],
                                            DEVICE=DEVICE
                                        )



print(f"Best loss {best_valid_loss}")