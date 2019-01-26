#Basic usage: python predict.py flowers/test/53/image_03677.jpg checkpoint.pth  --gpu
#Options:
#Return top KK most likely classes: python predict.py input checkpoint --top_k 3
#Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
#Use GPU for inference: python predict.py input checkpoint --gpu
import argparse
import os
import numpy as np
import pandas
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import json

from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image


def main():
    args = get_arguments()
    cuda = args.cuda
    model = load_checkpoint(args.checkpoint, cuda)
    model.idx_to_class = dict([[v,k] for k, v in model.class_to_idx.items()])
    
    with open(args.categories, 'r') as f:
        cat_to_name = json.load(f)
      
    prob, classes = predict(args.input, model, args.cuda, topk=int(args.top_k))
    print([cat_to_name[x] for x in classes])
    
    
def get_arguments():
    """ 
    Retrieve command line keyword arguments
    """
    parser_msg = 'Predict.py takes 2 manditory command line arguments, \n\t1.The image to have a predition made and \n\t2. the checkpoint from the trained nerual network'
    parser = argparse.ArgumentParser(description = parser_msg)

    # Manditory arguments
    parser.add_argument("input", action="store")
    parser.add_argument("checkpoint", action="store")

    # Optional arguments
    parser.add_argument("--top_k", action="store", dest="top_k", default=5, help="Number of top results you want to view.")
    parser.add_argument("--category_names", action="store", dest="categories", default="cat_to_name.json", 
                        help="Number of top results you want to view.")
    parser.add_argument("--gpu", action="store_true", dest="cuda", default=False, help="Set Cuda True for using the GPU")

    return parser.parse_args()

        
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    image_ratio = image.size[1] / image.size[0]
    image = image.resize((256, int(image_ratio*256)))
    half_the_width = image.size[0] / 2
    half_the_height = image.size[1] / 2
    image = image.crop((half_the_width - 112,
                       half_the_height - 112,
                       half_the_width + 112,
                       half_the_height + 112))
    
    image = np.array(image)
    image = image/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])
    
    image = (image - mean) / std_dev
    image = image.transpose((2, 0, 1))
    
    return torch.from_numpy(image)

    
def predict(image_path, model, cuda, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    
    image = Image.open(image_path)
    tensor = process_image(image)

    # Use GPU if available
    if cuda:
        with torch.no_grad():
            var_inputs = Variable(tensor.float().cuda())
    else: 
        with torch.no_grad():
            var_inputs = Variable(tensor)
            
    # Model is expecting 4d tensor, add another dimension
    var_inputs = var_inputs.unsqueeze(0)

    # Run image through model
    output = model.forward(var_inputs)  
   
    # Model's output is log-softmax,
    # take exponential to get the probabilities
    ps = torch.exp(output).data.topk(topk)
       
    # Move results to CPU if needed
    probs = ps[0].cpu() if cuda else ps[0]
    classes = ps[1].cpu() if cuda else ps[1]
           
    # Map classes to indices 
    inverted_class_to_idx = {model.class_to_idx[k]: k for k in model.class_to_idx}
    
    mapped_classes = list()
    for label in classes.numpy()[0]:
        mapped_classes.append(inverted_class_to_idx[label])
     
    # Return results
    return probs.numpy()[0], mapped_classes

def load_checkpoint(filepath, cuda):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
    
    for param in model.parameters():
        param.requires_grad = False    
    return model
main()