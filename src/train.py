import argparse
from model import Coord_VAE, CATH_dset, VAE_loss, Coord_VAE_seq_small
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

def train(param_dict, epoch, model_path):
    model = param_dict['model'].train()
    running_loss = []
    pbar = tqdm(total = len(param_dict['train_loader']))
    device = param_dict['device']
    loss_fn = param_dict['loss_fn']
    optimizer = param_dict['optim']
    step = param_dict['step']
    #writer = param_dict['tb_writer']
    train_loader = param_dict['train_loader']
    val_loader = param_dict['val_loader']
    
    total_loss = 0
    s = 0

    for coords in train_loader:
        optimizer.zero_grad()
        coords = coords.to(device)
        out, mu, logvar = model(coords)
        coords = coords.view(coords.size(0), coords.size(-1), 4, 3)
        loss = loss_fn(pred=out, gt=coords, mu=mu, logvar=logvar)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.update()
        s+=1
        pbar.set_description(str(epoch) + '/' + str(param_dict['train_epoch']) + ' ' + str(total_loss/s)[:6])
        writer.add_scalar('Loss/train', loss.item(), step)
    
    val_loss = 0
    s = 0
    model.eval()
    for coords in val_loader:
        coords = coords.to(device)
        out, mu, logvar = model(coords)
        coords = coords.view(coords.size(0), coords.size(-1), 4, 3)
        loss = loss_fn(out, coords, mu, logvar)
        val_loss += loss.item()
        s += 1
    pbar.set_description(str(epoch) + '/' + str(param_dict['train_epoch']) + ' ' + str(val_loss/s)[:6])
    writer.add_scalar('Loss/val', val_loss/s, epoch)

    model_states = {
        "epoch":epoch,
        "state_dict":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "loss":running_loss
    }

    torch.save(model_states, model_path)
    pbar.close()
    return val_loss/s, step




if __name__ == "__main__":
    print('------------Starting------------' + '\n')


    parser = argparse.ArgumentParser(
        description='Training script for EnNet')
    parser.add_argument(
        '--Device', type=str, help='CUDA device for training', default='0')

    args = parser.parse_args()
    train_set = CATH_dset.CATH_data(partition = 'train')
    val_set = CATH_dset.CATH_data( partition = 'validation')
    train_set = DataLoader(train_set, batch_size = 1, num_workers = 3, shuffle=True)
    val_set = DataLoader(val_set, batch_size = 1, num_workers = 3, shuffle = False)


    device = torch.device('cuda:'+ args.Device)
    model = Coord_VAE_seq_small.Coord_VAE_seq_small(device = device).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    loss_fn = VAE_loss.VAE_TFN_loss()
    model_out = '../trained_models/Coord_VAE/Coord_VAE_seq/Coord_VAE'

    param_dict = {
        'train_epoch': 100,
        'model': model,
        'optim': optimizer,
        'loss_fn': loss_fn,
        'train_loader': train_set,
        'val_loader': val_set,
        'device': device,
        'step': 0,
        'warmup': 4000,
        #'tb_writer': SummaryWriter(log_dir = 'runs/Base_VAE_dist+torsions')
    }

    print('Number of Training Sequence: ' + str(train_set.__len__()))
    print('Batch Size: ' + str(1))
    print('Learning Rate: ' + str(1e-3))
    print('Max training Epochs: ' + str(100))
    print('Early stopping patience: ' + str(15))
    print('Saving trained model at: ' + model_out)


    early_stop_cout = 0
    pre_val_loss = 100
    min_loss = 100
    for epoch in range(1, 100):
        model_path = model_out
        val_loss, step = train(param_dict=param_dict, epoch=epoch, model_path=model_path)
        param_dict['step'] = step
        if val_loss < min_loss:
            early_stop_cout = 0
            min_loss = val_loss
        else:
            early_stop_cout += 1
        #if early_stop_cout >= 15:
        #    break


