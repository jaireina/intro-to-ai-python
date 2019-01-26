import torch
import numpy as np
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import nn
from PIL import Image

from torch import optim
import torch.nn.functional as F
import json

devices = {
    'cuda' : "cuda:0",
    'cpu': "cpu"
}
    
def load_data(data_dir):
    ''' Arguments : data_dir where all the images are stored
        Returns : Dictionary with the training, test and validation dataloaders
    '''
    print("Loading data from: {}".format(data_dir))
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    valid_dir = data_dir + '/valid'

    data_transforms = {
        'train' : transforms.Compose([ transforms.RandomRotation(35), 
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]),

        'test' : transforms.Compose([ transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]),

        'validation' : transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    }

    
    image_datasets = {
        'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
        'validation': datasets.ImageFolder(valid_dir, transform=data_transforms['validation'])
    }

    dataloaders = {
        'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'test' : torch.utils.data.DataLoader(image_datasets['test'], batch_size=32),
        'validation' : torch.utils.data.DataLoader(image_datasets['validation'], batch_size=32)
    }
    
    print("Data has been loaded")
    return dataloaders, image_datasets

def get_model(arch = 'vgg16'):
    ''' Arguments : arch - architecture to be used 'vgg13' or 'vgg16', vgg16 is the default
        Returns : pretrained model
    '''
    if arch=='vgg13':
        #Download vgg13 model
        return models.vgg13(pretrained=True)
    else:
        #Download vgg16 model
        return models.vgg16(pretrained=True)
    
 
def define_model(class_to_indx = None, input_size=25088, output_size=102, hidden_layer1=400, hidden_layer2=200, dropout=0.5, arch='vgg16', learning_rate=0.001, use_gpu=False):
    ''' Arguments : defines the model that its going to be used with the input_size, output_size, hidden_layer1 inputs, hidden_layer2 inputs, dropout, architecture, learning_rate, and whether we want to use GPU or not in the use_gpu parameter
        Returns : model, criterion and optimizer configured with the parameters passed.
    '''
    model = get_model(arch)
    
    print("Configuring network")
          
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
        
    #create classifier and set it to be used in the model
    classifier = nn.Sequential(OrderedDict([
                              ('dropout', nn.Dropout(dropout)),
                              ('fc1', nn.Linear(input_size, hidden_layer1)),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_layer1, hidden_layer2)),
                              ('relu2', nn.ReLU()),
                              ('fc3', nn.Linear(hidden_layer2, output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    if(class_to_indx != None):
        model.class_to_idx = class_to_indx
    
    if use_gpu:
        model.to(devices['cuda'])
        
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    
    print("Network ready")
    
    return model, optimizer, criterion

#validation function
def validate(model,dataloader,criterion, use_gpu):
    if use_gpu:
        model.to(devices['cuda'])
    model.eval()
    validation_loss = 0
    accuracy = 0
    with torch.no_grad():
        for ii, (inputs,labels) in enumerate(dataloader):
            if use_gpu:
                inputs, labels = inputs.to(devices['cuda']), labels.to(devices['cuda'])
            output = model.forward(inputs)
            validation_loss += criterion(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
    return accuracy/len(dataloader), validation_loss/len(dataloader)

def train_network(model, optimizer, criterion, dataloaders , epochs = 5, use_gpu=False ):
    print("Starting training")
    print_every = 40
    steps = 0
    
    if use_gpu:
        print("Using GPU")
        model.to(devices['cuda'])
    
    for e in range(epochs):
        running_loss = 0

        for ii, (inputs, labels) in enumerate(dataloaders['train']):
            steps += 1
            if use_gpu:
                inputs, labels = inputs.to(devices['cuda']), labels.to(devices['cuda'])
                
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if steps % print_every == 0:
                #run validation
                accuracy, validation_loss  = validate(model, dataloaders['validation'], criterion, use_gpu)

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.3f} ".format(validation_loss),
                      "Validation Accuracy: {:.3f}".format(accuracy))

                running_loss = 0

def save_checkpoint(checkpoint_file_path, model, optimizer, arch, hidden_layer1, learning_rate, epochs):
    print("Saving checkpoint to {}".format(checkpoint_file_path))
    checkpoint = {
        'arch': arch,
        'hidden_layer1': 400,
        'class_to_indx': model.class_to_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs,
        'learning_rate': learning_rate
    }
    
    torch.save(checkpoint, checkpoint_file_path)

                
def load_checkpoint(path):
    print("Loading checkpoint from {}".format(path))
    conf = torch.load(path)

    model, optimizer, criterion = define_model(arch=conf['arch'], 
                                               hidden_layer1=conf['hidden_layer1'],
                                               class_to_indx=conf['class_to_indx'],
                                               learning_rate=float(conf['learning_rate'])
                                              )
    model.load_state_dict(conf['model_state_dict'])
    optimizer.load_state_dict(conf['optimizer_state_dict'])
       
    print("Checkpoint loaded")
    return model, optimizer, criterion

def predict(image_path, model, use_gpu=False, top_k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if use_gpu:
        model.to(devices['cuda'])
        
    img = process_image(image_path)
    img = torch.from_numpy(img)
    img = img.unsqueeze_(0)
    img = img.float()
    
    with torch.no_grad():
        if use_gpu:
            output = model.forward(img.cuda())
        else:
            output = model.forward(img)
            
    probabilities = F.softmax(output.data,dim=1)
    
    return probabilities.topk(top_k)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    
    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(244),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    pil_image = img_transforms(img)
    
    np_image = np.array(pil_image)
    
    return np_image


def get_category_name(cat_file, cat_number):
    '''
        Returns the name of a category given its number
    '''
    with open(cat_file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name[str(cat_number)]
