import numpy as np
import pandas as pd
import pickle5 as pickle
from tensorflow import keras

from efficientnet_pytorch import EfficientNet

import numpy as np
import pickle5 as pickle
import os

from tensorflow.keras import applications as efn
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, GlobalAveragePooling2D, MaxPool2D, UpSampling2D,Lambda,Dot,RepeatVector,Lambda
import tensorflow as tf


def convert_to_array(X_train):
  new_x = []
  for item in X_train:
    new_x.append(np.array(X_train[item])[0])

  return  np.array(new_x)

def get_model(input_shape, loss, optimizer, metric, epochs = 10, batch_size = 64):

    cnn_encoder = encoder(input_shape)

    input_l = Input(input_shape)
    input_r = Input(input_shape)

    X_1 = cnn_encoder(input_l)
    X_2 = cnn_encoder(input_r)

    X_1 = tf.keras.layers.GlobalMaxPool2D()(X_1)
    X_2 = tf.keras.layers.GlobalMaxPool2D()(X_2)

    cos_sim=Dot(axes=1, normalize=True, name='cos_sim')([X_1, X_2])

    sim_model = Model(inputs=[input_l,input_r], outputs=[cos_sim],name='sim_model')

    sim_model.compile(loss=loss, optimizer=optimizer, metrics=metric)

    return sim_model


def ConvertTabToImg(data_path_arr):
    j = 0
    for path in data_path_arr:
        inner_path, dirs, files = next(os.walk(path))
        file_count = len(files)
        print(file_count)
        i = 0
        data = {}
        for i in range(file_count):
            data[i] = []

        for filename in os.listdir(inner_path):
            # print(inner_path + filename)
            j += 1
            img = []
            with open(inner_path + filename) as f:
                # label = int(filename.split('_')[2])
                position = int(filename.split('_')[1])
                # print(position)
                for line in f:
                    row = []
                    for feature in line.split():
                        row.append(float(feature))
                    img.append(row)
                data[position].append(img)

                if 'str' in line:
                    break

    return data


def saveDataAsPickle(data, output_path):
    with open(output_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def loadDataFromPickle(path):
    with open(path, 'rb') as handle:
        imges = pickle.load(handle)
    return imges

def encoder(input_shape):
    base_model = efn.ResNet50(weights='imagenet', include_top=False, input_shape=(32,24,3))

    for layer in base_model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False

    model=Sequential()
    model.add(UpSampling2D(size=(1, 2)))
    model.add(base_model)

    return model

def processDataByImage():
    ## Convert Tabular to IMG Data
    NL_PATH = ['../IGTD/Result/Train_2/code/data/']
    CODE_PATH = ['../IGTD/Result/Train_1/nl/data/']
    data = ConvertTabToImg(CODE_PATH)

    ## Save The Data of CODE
    path_code_saved = './Data/pickle'
    data_file_name = '/x_images_code_train.pickle'
    saveDataAsPickle(data, path_code_saved + data_file_name)

    ## Save The Data of NL
    data = ConvertTabToImg(NL_PATH)
    data_file_name = '/x_images_nl_train.pickle'
    saveDataAsPickle(data, path_code_saved + data_file_name)

def train():
    path_nl_saved = './Data/pickle'
    path_code_saved = './Data/pickle'
    path_model_saved = './Data/models/model2'
    path_train = './Data/cosqa-train1.csv'

    train = pd.read_csv(path_train)

    with open(path_nl_saved + '/x_images_nl_train.pickle', 'rb') as handle:
        imges_l = pickle.load(handle)

    with open(path_code_saved + '/x_images_code_train.pickle', 'rb') as handle:
        imges_r = pickle.load(handle)

    y_train = train['label']
    # imges_l = np.repeat(imges_l[..., np.newaxis], 3, -1)
    # imges_r = np.repeat(imges_r[..., np.newaxis], 3, -1)

    imges_l = convert_to_array(imges_l)
    imges_r = convert_to_array(imges_r)

    imges_l = np.repeat(imges_l[..., np.newaxis], 3, -1)
    imges_r = np.repeat(imges_r[..., np.newaxis], 3, -1)

    input_shape = (32, 24, 3)
    loss = 'binary_crossentropy'
    optimizer = 'adam'
    metric = ['accuracy']

    print(len(imges_l))
    print(len(imges_r))

    model = get_model(input_shape, loss, optimizer, metric)

    model.fit([imges_l, imges_r], y_train, epochs=50, batch_size=64)

    model.save(path_model_saved)

