import csv
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dropout, Dense, Activation
# 二维卷积，以及二维池化 以及扁平化
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

class finalprobleam(object):
    """docstring for finalprobleam.
        仰望星空的人
    """

    def __init__(self,arg):
        super(finalprobleam, self).__init__()
        self.arg = arg

    def get_model(self):
        model = Sequential()                #CNN构建
        model.add(Convolution2D(
            input_shape=(512, 512, 1),
            #input_shape=(1, Width, Height),
            filters=8,
            kernel_size=3,
            strides=1,
            padding='same',
            data_format='channels_last',
        ))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(
            pool_size=2,
            strides=2,
            data_format='channels_last',
        ))
        model.add(Convolution2D(16, 3, strides=1, padding='same', data_format='channels_last'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(2, 2, data_format='channels_last'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(2, activation='softmax'))

        model.compile(optimizer=Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def load_train_data(self,check,file_path):
        print('-' * 30)
        print('load train images...')
        print('-' * 30)
        imgs_train = np.load(file_path+"/imgs_train.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_train /= 255
        print(len(imgs_train))
        if check==0:
            a=np.zeros(len(imgs_train))
        else:
            a=np.ones(len(imgs_train))
        imgs_mask_train = np_utils.to_categorical(a,num_classes=2)
        return imgs_train, imgs_mask_train

    def train(self,check,file_path):
        imgs_train,imgs_mask_train = self.load_train_data(check,file_path)
        model = self.get_model()
        model.fit(imgs_train,imgs_mask_train, epochs=5, batch_size=4,)

        model_checkpoint = ModelCheckpoint('final.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(imgs_train, imgs_mask_train, batch_size=4, nb_epoch=5, verbose=1, validation_split=0.2, shuffle=True,
                  callbacks=[model_checkpoint])

if __name__ == '__main__':

    file_name='file.csv'
    file = 'train/'
    check=0
    with open(file_name) as f:
        reader = csv.reader(f)

        finpro = finalprobleam("12")
        for row in reader:
            file_path=file+row[2]
            print(file_path,"\n")

            if '-'==row[3]:
                check=0
            else:
                check=1
            finpro.train(check,file_path)
