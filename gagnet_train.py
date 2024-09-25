#import argparse
import os
import torch
from glob import glob
import pickle
from models.utils import create_enhancement_model_and_configs
from recipes.utils import load_model_config
from tools.train_utils import run, gag_loss_fn
from dataio.dataio import (creat_data_pathes,
                            audio_data_loader,
                            evaluation_data_loader)
from models.utils import STFT, Enhancemet_gagnet_train


# use cuda if cuda available 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))

# construct the argument parser and parse the arguments

model_name = "gagnet"#args["name_of_enhancement_model"]
#BASEPATH = args["path_of_data_files"]
data_cfg = "./recipes/dataio/dataio.json" #args["data_config_path"]
train_cfg = "./recipes/training/training_gagnet.json" #args["train_config_path"]
model_cfg = "./recipes/models/gagnet.json"
# DATA PATHS
base_data_pth = "/home/dllabsharif/Documents/DATASETS"

base_clean_pth = os.path.join(base_data_pth,"clean_dataset")
base_noise_pth = os.path.join(base_data_pth,"Noise_dataset")
base_rever_pth = os.path.join(base_data_pth,"Noise_dataset/RIRS_NOISES")

clean_groups = ["CommonVoice", "60 Giga", "TIMIT"]
clean_scalse = [1,1,10]

noise_groups = ["QUT", "pointsource_noises", "musan_noise", "Audioset"]
noise_scalse = [1,5,10,2]

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

base_valid_pth = os.path.join(base_data_pth,"evaluation_dataset/enhancement/Validation_DATASET")

evalution_datasets = clean_groups

eval_filenames = []
for data_name in evalution_datasets:
    eval_filenames += glob(os.path.join(base_valid_pth,data_name,"*.mp3"))

for i in range(len(eval_filenames)):
    eval_filenames[i] = eval_filenames[i].split("/")[-1]


with open(os.path.join(base_valid_pth, "information.txt"), 'rb') as handle:
            dict_pth = pickle.load(handle)

print(f"number of valuation data : {len(eval_filenames)}")

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

valid_dataset = evaluation_data_loader(base_clean_pth,
                                      base_valid_pth,
                                      eval_filenames,
                                      dict_pth,
                                      data_configs["evaluation"]["SAMPLE_RATE"],
                                      data_configs["evaluation"]["BATCH_SIZE"], 
                                      data_configs["evaluation"]["NUM_WORKER"], 
                                      data_configs["evaluation"]["PIN_MEMORY"])

training_configs = load_model_config(train_cfg)


stft_layer = STFT().to(DEVICE)

model_configs = load_model_config(model_cfg)

enhacement, model_configs = create_enhancement_model_and_configs(model_configs = model_configs,
                                                                  DEVICE = DEVICE)

loss_fn = gag_loss_fn

if training_configs["IS_FINETUNE"]:
    enhacement.load_state_dict(torch.load(model_configs["param_save_path"]))

enh_pipline = Enhancemet_gagnet_train(enhacement, stft_layer)


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