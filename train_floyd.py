from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, Dense, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, load_img
from numpy import array
from keras import regularizers
import cv2
import pandas as pd
import numpy as np

path_output = r"/output/"
path_input = r"/data/"

def get_model():
    #init the model
    model= Sequential()
    #add conv layers and pooling layers 
    model.add(Convolution2D(32,3,3, input_shape=(50,50,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(32,3,3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(32,3,3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    #Now two hidden(dense) layers:
    model.add(Dense(output_dim = 500, activation = 'relu',
                    kernel_regularizer=regularizers.l2(0.008)
                    ))
    #output layer
    model.add(Dense(output_dim = 2))
    #Now copile it
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])
    return model
    

def load_data():
    image_path = path_input+"eye_images/"
    labels = path_input+"cursor_data.csv"
    X = []
    y = []
    df = pd.read_csv(labels)
    labels = df.iloc[:, [1, 2]].values
    for label in labels:
        y.append(list(label))
    # now images
   
    for i in range(len(y)):
        img = cv2.imread(image_path+"eye_{}.jpg".format(i), 0)
        #img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
        
        img = list(img.flatten())
        X.append(img)
    randomize = np.arange(len(y))
    np.random.shuffle(randomize)
    X = np.array(X)[randomize]
    y = np.array(y)[randomize]
    y[:,0] = y[:,0]
    y[:,1] = y[:,1]
    X = X.reshape(X.shape[0], 50, 50, 1)
    X = X/255.
    return X, y
    


model = get_model()
X, y = load_data()
#finally, start training
model.fit(X,y,
           nb_epoch = 200,
           batch_size = 64,
           validation_split=0.03
            )



#saving the weights
model.save_weights(path_output+"weights.hdf5",overwrite=True)

#saving the model itself in json format:
model_json = model.to_json()
with open(path_output+"model.json", "w") as model_file:
    model_file.write(model_json)
print("Model has been saved.")


