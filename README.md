# Developing an AI application

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, we might want to include an image classifier in a smart phone app. To do this, you&#39;d use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, you&#39;ll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at.

### **Install**

This project requires  **Python 3.x**  and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

We will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.

### Code

# Part 1-Developing an Image Classifier with Deep Learning

### In this first part of the project, you&#39;ll work through a Jupyter notebook to implement an image classifier with PyTorch.

The project is broken down into multiple steps:

- Load and preprocess the image dataset
- Train the image classifier on wer dataset
- Use the trained classifier to predict image content

- Load and preprocess the image dataset

Here I have use torchvision to load the dataThe dataset is split into three parts, training, validation, and testing. For the training, I have applied transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. I have make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.

The pre-trained networks I have used were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets I have normalized the means and standard deviations of the images to what the network expects. For the means, it&#39;s [0.485, 0.456, 0.406] and for the standard deviations [0.229, 0.224, 0.225], calculated from the ImageNet images. These values will shift each color channel to be centered at 0 and range from -1 to 1.

##               Image Preprocessing[¶](file:///tmp/AppData/Local/Temp/7zO4C158A53/Image%20Classifier%20Project.html#Image-Preprocessing)

              For it I have  use PIL to load the image.

First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [thumbnail](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [resize](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then we have to crop out the center 224x224 portion of the image.

Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. We&#39;ll need to convert the values. It&#39;s easiest with a Numpy array, which we can get from a PIL image like so np\_image = np.array(pil\_image).

As before, the network expects the images to be normalized in a specific way. For the means, it&#39;s [0.485, 0.456, 0.406] and for the standard deviations [0.229, 0.224, 0.225]. We&#39;ll want to subtract the means from each color channel, then divide by the standard deviation.

And finally, PyTorch expects the color channel to be the first dimension but it&#39;s the third dimension in the PIL image and Numpy array. We can reorder dimensions using [ndarray.transpose](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

- Train the image classifier on wer dataset

Now that the data is ready, it&#39;s time to build and train the classifier. As usual, I have used one of the pretrained models from torchvision.models to get the image features. Build and train a new feed-forward classifier using those features.



- Use the trained classifier to predict image content

               Once We get images in the correct format, it&#39;s time to write a function for making predictions with our model. A common practice is to predict the top 5 or so (usually called top-KK) most probable classes. You&#39;ll want to calculate the class probabilities then find the KK largest values.

To get the top KK largest values in a tensor use [x.topk(k)](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest k probabilities and the indices of those probabilities corresponding to the classes. We need to convert from these indices to the actual class labels using class\_to\_idx which hopefully we added to the model or from an ImageFolder we used to load the data. Make sure to invert the dictionary so we get a mapping from index to class as well.

Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.

# Part 2 - Building the command line application

Now that we&#39;ve built and trained a deep neural network on the flower data set, it&#39;s time to convert it into an application that others can use. Our application should be a pair of Python scripts that run from the command line. For testing, we should use the checkpoint you saved in the first part.

### Specifications

We have include two files train.py and predict.py. The first file, train.py, will train a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image.

- Train a new network on a data set with train.py
  - Basic usage: python train.py data\_directory
  - Prints out training loss, validation loss, and validation accuracy as the network trains
  - Options:
    - Set directory to save checkpoints: python train.py data\_dir --save\_dir save\_directory
    - Choose architecture: python train.py data\_dir --arch &quot;vgg13&quot;
    - Set hyperparameters: python train.py data\_dir --learning\_rate 0.01 --hidden\_units 512 --epochs 20
    - Use GPU for training: python train.py data\_dir --gpu
- Predict flower name from an image with predict.py along with the probability of that name. That is, we&#39;ll pass in a single image /path/to/image and return the flower name and class probability.
  - Basic usage: python predict.py /path/to/image checkpoint
  - Options:
    - Return top K_K_ most likely classes: python predict.py input checkpoint --top\_k 3
    - Use a mapping of categories to real names: python predict.py input checkpoint --category\_names cat\_to\_name.json
    - Use GPU for inference: python predict.py input checkpoint --gpu
