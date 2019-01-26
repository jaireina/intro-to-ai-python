''' 
    Sample usage:
        python train.py ./flowers --gpu
'''

import argparse
import utilities
import torch

ap = argparse.ArgumentParser()
ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./terminal_checkpoint.pth")

ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type=str)
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--hidden_units', dest="hidden_units", action="store", default=400, type=int)
ap.add_argument('--epochs', dest="epochs", action="store", default=1, type=int)

ap.add_argument('--gpu', dest="gpu", action="store_true")

arguments = ap.parse_args()
data_dir = arguments.data_dir[0]
save_dir = arguments.save_dir
learning_rate = arguments.learning_rate
arch = arguments.arch
hidden_units = arguments.hidden_units
use_gpu = arguments.gpu and torch.cuda.is_available()
epochs = arguments.epochs
dataloaders, image_datasets = utilities.load_data(data_dir)

model, optimizer, criterion = utilities.define_model(arch=arch, hidden_layer1=hidden_units, use_gpu=use_gpu )
utilities.train_network(model, optimizer, criterion, epochs=epochs, dataloaders=dataloaders, use_gpu=use_gpu)

model.class_to_idx = image_datasets['train'].class_to_idx
utilities.save_checkpoint(save_dir, model, optimizer, arch, hidden_units, learning_rate, epochs )

print("Network has finished its training")