import numpy as np
from PIL import Image
from keras.models import load_model

def save_model_to_disk(model):
    """ Take in a model and save it along with its weights to disk """
    print("Saving model to disk")
    
    model.save('model.h5')
    
    print("Saved model to disk")

def load_model_from_disk():
    print("Loading model from disk")

    model = load_model('model.h5')

    print("Loaded model from disk")

    model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    return model

def img_to_matrix(filename, img_rows, img_cols):
    """
    takes a filename and turns it into a numpy array of greyscale pixels
    """
    img = Image.open(filename).convert('L')
    img = img.resize((img_rows, img_cols))
    img = list(img.getdata())
    img = np.array(img)
    img = img.reshape(img_rows, img_cols, 1)
    return img