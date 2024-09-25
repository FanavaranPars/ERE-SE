import time
from tqdm import tqdm
import pickle
import os

import torch
import torch.nn.functional as F
import torch.nn as nn



def dbaiat_loss_fn(esti, label, frame_num):
    """Prepare DB-AIAT loss for training
    
    Arguments
    ---------
        esti_list : float (Tensor)
            Output of each defined layer of DB-AIAT for computing enhancement loss.
        label : float (Tensor)
            Pre-processed STFT of the clean input as a target for computing enhancement loss.
        frame_list : int
            Number of label frame.

    Returns
    -------
        loss : float (Tensor)
            The DB-AIAT loss.

    """

    mask_for_loss = []
    utt_num = esti.size()[0]
    with torch.no_grad():
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_num[i], esti.size()[-1]), dtype=esti.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
        com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
    mag_esti, mag_label = torch.norm(esti, dim=1), torch.norm(label, dim=1)
    loss1 = (((esti - label) * com_mask_for_loss) ** 2).sum() / com_mask_for_loss.sum()
    loss2 = (((mag_esti - mag_label) * mask_for_loss) ** 2).sum() / mask_for_loss.sum()
    return 0.5 * (loss1 + loss2)


def train_epoch(dataset, model, optimizer, loss_fn, feat_type='sqrt', grad_acc_step=1, step_show=100,\
                 DEVICE = 'cuda'):
    """train each epoch

    Arguments
    ---------
    dataset : class
        Training dataset

    model : class
        Model for training

    optimizer : function
        Training optimizer
        
    loss_fn : function
        Loss function

    grad_acc_step : int
        number of iteration to update parameters.

    step_show : int
        Number of batches to reduce learning rate and show training results

    DEVICE : str
        CPU or GPU.

    
    Returns
    -------
    total_loss : float
        Train loss for the epoch
    """
    model.train()

    total_loss = 0
    loss_section = 0
    section = 1

    counter = 0
    ex_counter = 0
    torch.cuda.empty_cache()
    start = time.time()
    for noisy_input, clean_target, length_ratio in tqdm(dataset):
        
        length_ratio = length_ratio.to(DEVICE)
        noisy_input = noisy_input.to(DEVICE)
        clean_target = clean_target.to(DEVICE)

        esti_list, target_stft, frame_num = model(noisy_input, clean_target, length_ratio)
        loss = loss_fn(esti_list, target_stft, frame_num)

        loss.backward()

        total_loss += loss.detach().cpu().item()
        counter += 1

        # graph is cleared here
        if counter % grad_acc_step == 0:
            optimizer.step()
            optimizer.zero_grad()


        if counter  % step_show == 0:
            finish = time.time()

            lr = optimizer.param_groups[0]['lr']
            l = (total_loss - loss_section) / (counter - ex_counter)
            print(f"Section {section}. lr: {lr:.5f}, Loss: {l:.5f}, Time (Min): {round((finish - start) / 60, 3)}")

            loss_section = total_loss
            ex_counter = counter
            section += 1
            start = time.time()

    optimizer.zero_grad()


    total_loss = total_loss / counter
    print(f"Total Train Loss: {total_loss:.5f}")

    return total_loss

def evaluate_epoch(dataset, model, loss_fn, feat_type='sqrt', DEVICE = 'cuda'):
    """Evaluate model with loss

    Arguments
    ---------
    dataset : class
        Training dataset

    model : class
        Model for training
        
    loss_fn : function
        Loss function

    DEVICE : str
        CPU or GPU.

    
    Returns
    -------
    total_loss : float
        Train loss for the epoch
    """
    model.eval()

    total_loss = 0
    counter = 0
    torch.cuda.empty_cache()
    with torch.no_grad():  
        for noisy_input, clean_target, length_ratio in tqdm(dataset):
            length_ratio = length_ratio.to(DEVICE)
            noisy_input = noisy_input.to(DEVICE)
            clean_target = clean_target.to(DEVICE)

            esti_list, target_stft, frame_num = model(noisy_input, clean_target, length_ratio)
            loss = loss_fn(esti_list, target_stft, frame_num)

            total_loss += loss.detach().cpu().item()
            counter += 1

    total_loss = total_loss / counter

    return total_loss


# run the training and evaluation.
def run(model,
        train_loader,
        validation_loader,
        optimizer,
        loss_fn,
        save_model_path,
        chkp_path,
        step_show,
        n_epoch,
        grad_acc_step=1,
        is_finetune = False,
        DEVICE = 'cuda'
        ):
    """execuation of training, evaluating and saving best model

    Arguments
    ---------
    model : class
        Model for training
        
    train_loader : class
        Training data loader
        
    validation_loader : class
        Validation data loader

    optimizer : function
        Training optimizer
        
    loss_fn : function
        Loss function
        
    save_model_path : str
        Path for saving model parameters

    step_show : int
        Number of batches to reduce learning rate and show training results

    n_epoch : str
        Number of epoches

    grad_acc_step : int
        number of iteration to update parameters.

    DEVICE : str
        CPU or GPU.

    Returns
    -------
    best_loss : float
        Best alidation loss.
        
    """
    
    val_loss = []
    train_loss = []
    
    if is_finetune:
        best_loss = evaluate_epoch(validation_loader,
                                    model, loss_fn,
                                    DEVICE=DEVICE)
        print(f'FineTuning Loss Starts from : {best_loss}')
    else: 
        best_loss = 1e10
    
    best_epoch = 0
    for epoch in range(n_epoch):
        start = time.time()
        print('\n',f"--- start epoch {epoch+1} ---")
        
        train_loss.append(train_epoch(train_loader, model,
                                        optimizer, loss_fn, 
                                        grad_acc_step, 
                                       step_show, DEVICE=DEVICE))
        
        val_loss.append(evaluate_epoch(validation_loader,
                                         model, loss_fn,
                                           DEVICE=DEVICE))
        
        finish = time.time()

        print(f"Train_Loss: {train_loss[-1]:.5f}") 
        print(f"Val_Loss: {val_loss[-1]:.5f}")
        print(f"Epoch_Time (min): {round((finish - start) / 60, 3)}")

        # save best model
        if val_loss[-1] < best_loss:
            best_epoch = epoch+1
            best_loss = val_loss[-1]      
            torch.save(model.enhancement.state_dict(), save_model_path)

        result_dict = {"train_losses": train_loss,
                       "vall_losses": val_loss,
                       "best_valid_loss": best_loss,
                       "best_epoch" : best_epoch,
                       "n_f_epoch" : epoch+1,
                       "finetuning": is_finetune
                       }
        
        with open(os.path.join(chkp_path), 'wb') as handle:
            pickle.dump(result_dict, handle)

    return train_loss, val_loss, best_loss