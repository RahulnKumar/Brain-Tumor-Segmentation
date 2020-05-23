# Imports

from keras.models import Sequential,load_model,Model,model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D,concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Input, merge, UpSampling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras import backend as K

# Initializations
smooth = 1
img_size = 120
no_of_channels = 1


# Creating Model    
def simple_model():
    model = Sequential()
    #add layer1
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(no_of_channels,img_size,img_size) ,padding='same'))
    #add layer2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    #add layer3
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    #add layer4
    model.add(MaxPooling2D(pool_size=(3, 3), strides = (2, 2)))
    model.add(Dropout(0.1))
    #add layer5
    #model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    #add layer6
    #model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    #add layer7
    #model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    #add layer8
    #model.add(MaxPooling2D(pool_size=(3, 3), strides = (2, 2)))
    #model.add(Dropout(0.5))
    #add layer9
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    #add layer10
    #model.add(Flatten())
    #model.add(Dense(256, activation='relu'))
    #add layer11
    #model.add(Flatten())
    model.add(Dense(14400, activation='sigmoid'))        # output fully connected layer is set to img_size*img_size = 120*120 = 14400 units

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=dice_coef_loss, optimizer=sgd, metrics = [dice_coef])
    
    return model


# Metrics
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)