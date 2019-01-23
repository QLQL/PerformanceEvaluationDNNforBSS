from keras import layers, models, optimizers, initializers
from keras import backend as K
import tensorflow as tf

import numpy as np
from keras.regularizers import l2



def preprocess():
    inp_shape = (100, 127 * 3)
    out_shape = (100, 127)

    inp = layers.Input(shape=inp_shape, name='input')

    def generateCovLayer():
        convModel = models.Sequential(name='conv_layer')
        convModel.add(layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 2), padding='same', input_shape=(100, 127, 3),name='conv1'))
        convModel.add(layers.BatchNormalization())
        convModel.add(layers.Activation('relu'))
        convModel.add(layers.pooling.MaxPooling2D(pool_size=(1, 2), padding='valid'))

        # convModel.add( layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(1, 2), padding='same',name='conv2'))
        # convModel.add(layers.BatchNormalization())
        # convModel.add(layers.Activation('relu'))
        return convModel
    convModel = generateCovLayer()

    def presplit(input):
        # inp_shape = (100, 127 * 3)
        inputR = K.stack(tf.split(input, 3, axis=2), axis=-1)  # (100, 127, 3)

        return inputR
    inputR = layers.Lambda(presplit, name='reshape1')(inp)  # (100, 127, 3)

    convoutput = convModel(inputR)  # (100,32,64) (time, frequency, #kernles)

    return [inp_shape, out_shape, inp, convoutput]




def GenerateBLSTMTime():
    #################################### The DNN input
    inp_shape = (100, 127 * 3)
    out_shape = (100, 127)

    inp = layers.Input(shape=inp_shape, name='input')


    #################################### RNN along time

    SIZE_RLAYERS = 256
    # Regularization parameters
    DROPOUT = 0.5  # Feed forward dropout
    RDROPOUT = 0.2  # Recurrent dropout
    L2R = 1e-6  # L2 regularization factor

    x = inp

    for i in range(2): # two stacked BiLSTM
        x = layers.Bidirectional(layers.LSTM(SIZE_RLAYERS, return_sequences=True,
                               kernel_regularizer=l2(L2R),
                               recurrent_regularizer=l2(L2R),
                               bias_regularizer=l2(L2R),
                               dropout=DROPOUT,
                               recurrent_dropout=RDROPOUT))(x)
    mask_o = layers.TimeDistributed(layers.Dense(out_shape[-1],
                                     activation='sigmoid',
                                     kernel_regularizer=l2(L2R),
                                     bias_regularizer=l2(L2R)),
                               name='mask_o')(x)

    train_model = models.Model(inputs=[inp], outputs=[mask_o])

    return train_model

def GenerateBLSTMFrequency():
    #################################### The DNN input
    inp_shape = (100, 127 * 3)
    out_shape = (100, 127)

    inp = layers.Input(shape=inp_shape, name='input')

    #################################### RNN along Frequency

    def presplit(inp):
        # inp_shape = (100, 127 * 3)
        inputR = K.stack(tf.split(inp, 3, axis=2), axis=-1)  # (100, 127, 3)
        # add one frequency bin, such relax band segmentation 127->128
        _, lastf = tf.split(inputR, [126, 1], axis=2) # (100, 1, 3)
        inputRC = K.concatenate([inputR,lastf],axis=2) # (100, 128, 3)

        # reshape again
        inputRCR = layers.Reshape((inp._keras_shape[1], 32, 12))(inputRC) #128 x 3 ----> 32 x 12

        return inputRCR

    inputR = layers.Lambda(presplit, name='reshape1')(inp)  # (100, 32, 12)


    SIZE_RLAYERS1 = 256
    # Regularization parameters
    DROPOUT1 = 0  # Feed forward dropout
    RDROPOUT1 = 0.2  # Recurrent dropout
    SIZE_RLAYERS2 = 256
    # Regularization parameters
    DROPOUT2 = 0.5  # Feed forward dropout
    RDROPOUT2 = 0.2  # Recurrent dropout
    L2R = 1e-6  # L2 regularization factor

    rnnModel = models.Sequential(name='BiLSTM_f')
    rnnModel.add(layers.Bidirectional(layers.LSTM(SIZE_RLAYERS1, return_sequences=True,
                                                  kernel_regularizer=l2(L2R),
                                                  recurrent_regularizer=l2(L2R),
                                                  bias_regularizer=l2(L2R),
                                                  dropout=DROPOUT1,
                                                  recurrent_dropout=RDROPOUT1),
                                      input_shape=[inputR._keras_shape[2], inputR._keras_shape[3]]))  # (32, 64) (frequency, #kernles)
    rnnModel.add(layers.Bidirectional(layers.LSTM(SIZE_RLAYERS2, return_sequences=True,
                                                  kernel_regularizer=l2(L2R),
                                                  recurrent_regularizer=l2(L2R),
                                                  bias_regularizer=l2(L2R),
                                                  dropout=DROPOUT2,
                                                  recurrent_dropout=RDROPOUT2)))

    rnnModel.add(layers.TimeDistributed(layers.Dense(4,  # 32*4~127
                                                     activation='sigmoid',
                                                     kernel_regularizer=l2(L2R),
                                                     bias_regularizer=l2(L2R)),
                                        name='mask_o'))


    x = layers.TimeDistributed(rnnModel, name='RNN_f')(inputR)

    def easyreshapesplit(x):
        #y = K.reshape(x, shape=[-1, x._keras_shape[1], np.prod(x._keras_shape[2::])])#(100, 128) (time, #freq)
        y = layers.Reshape((x._keras_shape[1], np.prod(x._keras_shape[2::])))(x)#(100, 128) (time, #freq)
        output, _ = tf.split(y, [127, 1], axis=2)
        return output

    y = layers.Lambda(easyreshapesplit, name='reshape2')(x)


    train_model = models.Model(inputs=[inp], outputs=[y])

    return train_model

def GenerateBLSTMTF():
    #################################### The DNN input
    inp_shape = (100, 127 * 3)
    out_shape = (100, 127)

    inp = layers.Input(shape=inp_shape, name='input')

    #################################### RNN along time

    SIZE_RLAYERS = 256
    # Regularization parameters
    DROPOUT = 0.5  # Feed forward dropout
    RDROPOUT = 0.2  # Recurrent dropout
    L2R = 1e-6  # L2 regularization factor

    rnnModel1 = models.Sequential(name='BiLSTM_t')
    rnnModel1.add(layers.Bidirectional(layers.LSTM(SIZE_RLAYERS, return_sequences=True,
                                                   kernel_regularizer=l2(L2R),
                                                   recurrent_regularizer=l2(L2R),
                                                   bias_regularizer=l2(L2R),
                                                   dropout=DROPOUT,
                                                   recurrent_dropout=RDROPOUT),
                                       input_shape=[inp._keras_shape[1],
                                                    inp._keras_shape[2]]))

    x1 = rnnModel1(inp)
    tmp1 = SIZE_RLAYERS * 2 / 128
    y_t = layers.Reshape((100, 128, tmp1))(x1)

    #################################### RNN along Frequency

    def presplit(inp):
        # inp_shape = (100, 127 * 3)
        inputR = K.stack(tf.split(inp, 3, axis=2), axis=-1)  # (100, 127, 3)
        # add one frequency bin, such relax band segmentation 127->128
        _, lastf = tf.split(inputR, [126, 1], axis=2)  # (100, 1, 3)
        inputRC = K.concatenate([inputR, lastf], axis=2)  # (100, 128, 3)

        # reshape again
        inputRCR = layers.Reshape((inp._keras_shape[1], 32, 12))(inputRC)  # 128 x 3 ----> 32 x 12

        return inputRCR

    inputR = layers.Lambda(presplit, name='reshape1')(inp)  # (100, 128, 3)

    SIZE_RLAYERS1 = 64
    # Regularization parameters
    DROPOUT1 = 0  # Feed forward dropout
    RDROPOUT1 = 0.2  # Recurrent dropout
    L2R = 1e-6  # L2 regularization factor

    rnnModel2 = models.Sequential(name='BiLSTM_f')
    rnnModel2.add(layers.Bidirectional(layers.LSTM(SIZE_RLAYERS1, return_sequences=True,
                                                  kernel_regularizer=l2(L2R),
                                                  recurrent_regularizer=l2(L2R),
                                                  bias_regularizer=l2(L2R),
                                                  dropout=DROPOUT1,
                                                  recurrent_dropout=RDROPOUT1),
                                      input_shape=[inputR._keras_shape[2],
                                                   inputR._keras_shape[3]]))  # (32, 64) (frequency, #kernles)
    rnnModel2.add(layers.TimeDistributed(layers.Dense(16,  # 32*4~127
                                                     activation='sigmoid',
                                                     kernel_regularizer=l2(L2R),
                                                     bias_regularizer=l2(L2R))))


    x = layers.TimeDistributed(rnnModel2, name='RNN_f')(inputR)
    tmp = np.prod(x._keras_shape[2::])/128
    y_f = layers.Reshape((x._keras_shape[1], 128, tmp))(x)  # (100, 128)

    y = layers.concatenate([y_t,y_f],axis=-1)

    # post processing with more conv layers
    zz1 = layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same')(y)
    z1 = layers.BatchNormalization()(zz1)
    z1 = layers.LeakyReLU()(z1)

    # zz2 = layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same')(z1)
    # z2 = layers.BatchNormalization()(zz2)
    # z2 = layers.LeakyReLU()(z2)
    #
    # zz3 = layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same')(z2)
    # z3sum = layers.Add()([zz1, zz3])
    # z3 = layers.BatchNormalization()(z3sum)
    # z3 = layers.LeakyReLU()(z3)

    y_o = layers.Conv2D(filters=1, kernel_size=(5, 5), padding='same')(z1)
    y_o = layers.Activation('sigmoid')(y_o)

    def easyreshapesplit(x):
        #y = K.reshape(x, shape=[-1, x._keras_shape[1], np.prod(x._keras_shape[2::])])#(100, 128) (time, #freq)
        y = K.squeeze(x,axis=-1)#(100, 128) (time, #freq)
        output, _ = tf.split(y, [127, 1], axis=2)
        return output

    mask_o = layers.Lambda(easyreshapesplit, name='output')(y_o)


    train_model = models.Model(inputs=[inp], outputs=[mask_o])

    return train_model



if __name__=="__main__":
    train_model = GenerateBLSTMTF()
    print(train_model.summary())
    from keras.utils import plot_model
    plot_model(train_model, to_file='Model.png')

