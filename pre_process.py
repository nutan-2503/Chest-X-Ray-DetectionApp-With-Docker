import keras
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model, Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Input, Average
import numpy as np
import scipy

# Converting a given image into tensor
def path_to_tensor(data_path):
    # Loading image
    img = image.load_img(data_path, target_size=(256, 256))
    # Converting image to array
    img = image.img_to_array(img)
    # Converting to 4D tensor => (1, 256, 256, 3)
    return np.expand_dims(img, axis=0)

def getting_two_layer_weights():
    # Importing the model
    model = tf.keras.models.load_model("./weights/mobilenet.hdf5", compile=False)
    # Getting the AMP layer weight
    all_amp_layer_weights = model.layers[-1].get_weights()[0]
    # Extracting the wanted output
    mobilenet_model = Model(inputs = model.input, outputs = (model.layers[-5].output, model.layers[-1].output))
    return mobilenet_model, all_amp_layer_weights

# Grad-CAM function
def CAM_func(img_path, model, all_amp_layer_weights):
    # Getting filtered images from last convolutional layer + model prediction output
    last_conv_output, predictions = model.predict(path_to_tensor(img_path)) 
    # Eliminating dimensions of one    
    last_conv_output = np.squeeze(last_conv_output)
    # Model's prediction
    predicted_class = np.argmax(predictions)
    # Bilinear upsampling (resize each image to size of original image)
    mat_for_mult = scipy.ndimage.zoom(last_conv_output, (32, 32, 0.5), order = 1)  # dim from (16, 16, 1024) to (512, 512, 1024)
    # Getting the AMP layer weights
    amp_layer_weights = all_amp_layer_weights[:, predicted_class] # dim: (1024,)    
    # CAM for object class that is predicted to be in the image
    final_output = np.dot(mat_for_mult, amp_layer_weights) # dim: 512 x 512
    # Return class activation map (CAM)
    return final_output, predicted_class
