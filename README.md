# ERI-Speech-Enhancement

This is the repository of speech enhancement modules which have been prepared by the Electronic Research Institute(ERI).

Two models,DB-AIAT and GaGNet,have been trained and evaluated for the task of enhancement. Experimental results on Voice Bank + DEMAND
demonstrate that DB-AIAT and GaGNet yield state-of-the-art performance.

## Checkpoints
Download the checkpoints [here](https://huggingface.co/FanavaranPars/ERI-SE).

The following is a simple guide for training and evaluating these mentioned models:

 ## How to train
 ### DB-AIAT
 For continuing the train of DB-AIAT, it is needed to run a simple command below in the terminal:
  ```
  python dbaiat_train.py
  ```
 ### GaGNet
 For continuing the train of GaGNet, it is needed to run a simple command below in the terminal:
   ```
   python gagnet_train.py
   ```
## Inference
For testing the results and getting hearing test, put noisy samples in **Noisy_Samples** directory. Then, you are able to have the enhanced files in **Enhanced_Samples** directory.

You can see the results simply by running the command below:
```
  python enhance.py -model {model_name} -device {device}
```

You can choose your target model from ['dbaiat', 'gagnet']

**Notice**: The target device can be choosen from ['cuda','cpu']. It is set to 'cuda' in default. It is recommended to use 'cuda' (Do not set device in the input arguemnts).


### DB-AIAT
It is needed to run a simple command below in the terminal:
  ```
  python enhance.py -model dbaiat 
  ```
 ### GaGNet
 It is needed to run a simple command below in the terminal:
   ```
   python enhance.py -model gagnet
   ```


