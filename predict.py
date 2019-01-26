''' 
    Sample usage:
        python predict.py ./flowers/test/102/image_08004.jpg ./terminal_checkpoint.pth
        python predict.py ./flowers/test/16/image_06657.jpg ./terminal_checkpoint.pth
'''

import argparse
import utilities
import torch

ap = argparse.ArgumentParser()
ap.add_argument('base_input', nargs='*', action="store")
ap.add_argument('--top_k', dest="top_k", action="store", default=1, type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='./cat_to_name.json')
ap.add_argument('--gpu', dest="gpu", action="store_true")

arguments = ap.parse_args()
input_image = arguments.base_input[0]
checkpoint = arguments.base_input[1]
top_k = arguments.top_k 
category_names = arguments.category_names
use_gpu = arguments.gpu and torch.cuda.is_available()

model, optimizer, criterion = utilities.load_checkpoint(checkpoint)
results = utilities.predict(input_image, model, use_gpu=use_gpu, top_k=top_k)

if use_gpu:
    probabilities = results[0].cpu().numpy()[0]
else:
    probabilities = results[0].numpy()[0]

if use_gpu:
    names = results[1].cpu().numpy()[0]
else:
    names = results[1].numpy()[0]
    

print("----- Results: -------")
for i in range(top_k):
    print("{} with a probability of {:.2%}".format(utilities.get_category_name(category_names,  names[i]), probabilities[i]))

