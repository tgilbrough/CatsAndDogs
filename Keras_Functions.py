import numpy as np
from PIL import Image
from keras.models import model_from_json

def save_model(model):
    """ Take in a model and save it along with its weights to disk """
    # write model to json file
    model_json = model.to_json()

    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("model.h5")
    
    print("Saved model to disk")

def load_model():
    # load json and create model
    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights("model.h5")
    
    print("Loaded model from disk")

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

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