#!/usr/bin/env python
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.
# 
# Please make sure if you are running this notebook in the workspace that you have chosen GPU rather than CPU mode.

# In[2]:


# Imports here
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


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[3]:


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[4]:


#Setting out the batch size
batch_size = 64

# Since we have to transform data from test, valid and training we must need to create a set of those elements.
data_transforms = transforms.Compose([transforms.Resize(256), 
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Image dataserts
image_datasets = [datasets.ImageFolder(train_dir, transform=train_transforms),                      
                  datasets.ImageFolder(valid_dir, transform=data_transforms),                   
                  datasets.ImageFolder(test_dir, transform=data_transforms)]                        

# Data Loaders
dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size=batch_size, shuffle=True), 
               torch.utils.data.DataLoader(image_datasets[1], batch_size=batch_size),               
               torch.utils.data.DataLoader(image_datasets[2], batch_size=batch_size)]                        

# Validating the device available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#Validating the dataset sizes for validating that were succesufully loaded. 
print("Train" + str(len(dataloaders[0])))
print("Test" + str(len(dataloaders[1])))
print("Valid" + str(len(dataloaders[2])))


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[5]:


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
#Checking if the dataloaders works.
images, labels = next(iter(dataloaders[0]))
images.size()


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.

# In[21]:


model = models.densenet121(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
    
classifier = nn.Sequential(OrderedDict([('input',  nn.Linear(1024, 550)),
                                        ('relu1',  nn.ReLU()),
                                        ('dropout1',  nn.Dropout(0.5)),
                                        ('linear2',  nn.Linear(550, 200)),
                                        ('relu2',  nn.ReLU()),
                                        ('linear3', nn.Linear(200, 102)),
                                        ('output', nn.LogSoftmax(dim=1))
                                       ]))
model.classifier = classifier


# In[22]:


def train_model(criterion, optimizer, epochs=10, cuda=False):
    start_time = time.time()
    steps = 0
    print_every = 10
    
    # Checking if gpu available
    if cuda:
        model.cuda()
    else:
        model.cpu()

    # Epochs loop.
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(dataloaders[0]):
            inputs, labels = Variable(inputs), Variable(labels)
            steps += 1
            
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.data[0]
            
            if steps % print_every == 0:
                model.eval()
            
                accuracy = 0
                validation_loss = 0
            
                for ii, (images, labels) in enumerate(dataloaders[1]):
                    #Avoiding gradients
                    with torch.no_grad():
                        inputs = Variable(images)
                        labels = Variable(labels)
        
                    if cuda:
                        inputs, labels = inputs.cuda(), labels.cuda()
        
                    output = model.forward(inputs)
                    validation_loss += criterion(output, labels).data[0]
                    ps = torch.exp(output).data
                    equality = (labels.data == ps.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("[Stats] -> ",
                      "Epoch: {} / {}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(validation_loss/len(dataloaders[1])),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(dataloaders[1])))

                running_loss = 0
                model.train()
                
    elapsed_time = time.time() - start_time
    print('Elapsed Time: {:.0f}m {:.0f}s'.format(elapsed_time//60, elapsed_time % 60))

# Defining the learning rate and the number of epochs in order to improve model performance.
learning_rate = 0.001
epochs = 5

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

cuda = torch.cuda.is_available

#Calling the train model method in order to start training the model.
train_model(criterion=criterion, optimizer=optimizer, epochs=epochs, cuda=cuda)


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[13]:


def validate_model(cuda=False):
    model.eval()
    accuracy = 0
    # Validate if GPU available
    if cuda:
        model.cuda()
    else:
        model.cpu()

    for ii, (images, labels) in enumerate(dataloaders[2]):

        with torch.no_grad():
            inputs = Variable(images)
            labels = Variable(labels)

        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        output = model.forward(inputs)
        ps = torch.exp(output).data
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    print("Testing Accuracy: {:.3f}".format(accuracy/len(dataloaders[2])))


# In[14]:


cuda = torch.cuda.is_available
validate_model(cuda=cuda)


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[25]:


checkpoint = {'input_size': 1024,
              'output_size': 102,
              'epochs': epochs,
              'learning_rate':learning_rate,
              'batch_size': 64,
              'data_transforms': data_transforms,
              'model': models.densenet121(pretrained=True),
              'classifier': classifier,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': image_datasets[0].class_to_idx
             }

torch.save(checkpoint, 'checkpoint.pth')


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[6]:


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
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

model = load_checkpoint('checkpoint.pth')
model


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[7]:


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


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[8]:


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# In[9]:




with Image.open('flowers/test/53/image_03677.jpg') as image:
    plt.imshow(image)

with Image.open('flowers/test/53/image_03677.jpg') as image:
    imshow(process_image(image))


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[15]:


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Put model in inference mode
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


# In[18]:


image_path = test_dir + '/15/image_06351.jpg'
probs, classes = predict(image_path, model)

print(probs)
print(classes)


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[17]:


# TODO: Display an image along with the top 5 classes
def view_prediction(image_path, model):
    prob, classes = predict(image_path, model)
    classes = [cat_to_name[x] for x in classes]
    
    fig, (ax1, ax2) = plt.subplots(figsize=(6,7), nrows=2)
    with Image.open(image_path) as img:
        ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(cat_to_name[image_path.split('/')[2]])
    
    y_pos = np.arange(len(classes))
    ax2.barh(y_pos, prob)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(classes)
    ax2.invert_yaxis()

view_prediction(image_path, model)


# In[ ]:




