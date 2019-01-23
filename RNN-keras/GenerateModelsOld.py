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



# def GenerateBLSTMTime():
#     # The same pre-processing is appled to different DNN structures
#     [inp_shape, out_shape, inp, convoutput] = preprocess()
#
#
#     def easyreshape(x):
#         xR = K.reshape(x, shape=[-1, 100, np.prod(convoutput._keras_shape[2::])])
#         return xR
#
#     convoutputR = layers.Lambda(easyreshape, name='reshape2')(convoutput)
#
#
#     SIZE_RLAYERS = 128
#     # Regularization parameters
#     DROPOUT = 0.5  # Feed forward dropout
#     RDROPOUT = 0.2  # Recurrent dropout
#     L2R = 1e-6  # L2 regularization factor
#
#     #dimension reduction in each frame
#     simpleModel = models.Sequential(name='dense_layer')
#     # simpleModel.add(layers.Dropout(0.5, input_shape=(convoutputR._keras_shape[-1],)))
#     # simpleModel.add(layers.Dense(256, activation='relu', kernel_regularizer=l2(L2R), bias_regularizer=l2(L2R)))
#     simpleModel.add(layers.Dense(256, activation='relu', kernel_regularizer=l2(L2R), bias_regularizer=l2(L2R),input_shape=(convoutputR._keras_shape[-1],)))
#     simpleModel.add(layers.BatchNormalization())
#
#     x = layers.TimeDistributed(simpleModel,name='Dense')(convoutputR)
#
#     for i in range(2): # two stacked BiLSTM
#         x = layers.Bidirectional(layers.LSTM(SIZE_RLAYERS, return_sequences=True,
#                                kernel_regularizer=l2(L2R),
#                                recurrent_regularizer=l2(L2R),
#                                bias_regularizer=l2(L2R),
#                                dropout=DROPOUT,
#                                recurrent_dropout=RDROPOUT))(x)
#     mask_o = layers.TimeDistributed(layers.Dense(out_shape[-1],
#                                      activation='sigmoid',
#                                      kernel_regularizer=l2(L2R),
#                                      bias_regularizer=l2(L2R)),
#                                name='mask_o')(x)
#
#     train_model = models.Model(inputs=[inp], outputs=[mask_o])
#
#     return train_model

def GenerateBLSTMTime():
    # The same pre-processing is appled to different DNN structures
    [inp_shape, out_shape, inp, convoutput] = preprocess()


    def easyreshape(x):
        xR = K.reshape(x, shape=[-1, 100, np.prod(convoutput._keras_shape[2::])])
        return xR

    convoutputR = layers.Lambda(easyreshape, name='reshape2')(convoutput)


    SIZE_RLAYERS = 256
    # Regularization parameters
    DROPOUT = 0.5  # Feed forward dropout
    RDROPOUT = 0.2  # Recurrent dropout
    L2R = 1e-6  # L2 regularization factor

    # #dimension reduction in each frame
    # simpleModel = models.Sequential(name='dense_layer')
    # # simpleModel.add(layers.Dropout(0.5, input_shape=(convoutputR._keras_shape[-1],)))
    # # simpleModel.add(layers.Dense(256, activation='relu', kernel_regularizer=l2(L2R), bias_regularizer=l2(L2R)))
    # simpleModel.add(layers.Dense(256, activation='relu', kernel_regularizer=l2(L2R), bias_regularizer=l2(L2R),input_shape=(convoutputR._keras_shape[-1],)))
    # simpleModel.add(layers.BatchNormalization())
    #
    # x = layers.TimeDistributed(simpleModel,name='Dense')(convoutputR)

    x = convoutputR

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
    # The same pre-processing is appled to different DNN structures
    [inp_shape, out_shape, inp, convoutput] = preprocess()

    # The RNN model that being applied to each frame
    SIZE_RLAYERS = 128
    # Regularization parameters
    DROPOUT = 0.5  # Feed forward dropout
    RDROPOUT = 0.2  # Recurrent dropout
    L2R = 1e-6  # L2 regularization factor

    rnnModel = models.Sequential(name='BiLSTM_f')
    rnnModel.add(layers.Bidirectional(layers.LSTM(SIZE_RLAYERS, return_sequences=True,
                                             kernel_regularizer=l2(L2R),
                                             recurrent_regularizer=l2(L2R),
                                             bias_regularizer=l2(L2R),
                                             dropout=DROPOUT,
                                             recurrent_dropout=RDROPOUT),
                                      input_shape=[convoutput._keras_shape[2], convoutput._keras_shape[3]]))#(32, 64) (frequency, #kernles)
    rnnModel.add(layers.Bidirectional(layers.LSTM(SIZE_RLAYERS, return_sequences=True,
                                                  kernel_regularizer=l2(L2R),
                                                  recurrent_regularizer=l2(L2R),
                                                  bias_regularizer=l2(L2R),
                                                  dropout=DROPOUT,
                                                  recurrent_dropout=RDROPOUT)))

    rnnModel.add(layers.TimeDistributed(layers.Dense(4, # 32*4~127
                                                 activation='sigmoid',
                                                 kernel_regularizer=l2(L2R),
                                                 bias_regularizer=l2(L2R)),
                                    name='mask_o'))

    x = layers.TimeDistributed(rnnModel, name='RNN_f')(convoutput)

    def easyreshapesplit(x):
        #y = K.reshape(x, shape=[-1, x._keras_shape[1], np.prod(x._keras_shape[2::])])#(100, 128) (time, #freq)
        y = layers.Reshape((x._keras_shape[1], np.prod(x._keras_shape[2::])))(x)#(100, 128) (time, #freq)
        output, _ = tf.split(y, [127, 1], axis=2)
        return output

    y = layers.Lambda(easyreshapesplit, name='reshape3')(x)


    train_model = models.Model(inputs=[inp], outputs=[y])

    return train_model

# def GenerateBLSTMTF():
#     # The same pre-processing is appled to different DNN structures
#     [inp_shape, out_shape, inp, convoutput] = preprocess()
#
#     # The RNN model that being applied to each frame
#     SIZE_RLAYERS = 128
#     # Regularization parameters
#     DROPOUT = 0.5  # Feed forward dropout
#     RDROPOUT = 0.2  # Recurrent dropout
#     L2R = 1e-6  # L2 regularization factor
#
#     rnnModel = models.Sequential(name='BiLSTM_f')
#     rnnModel.add(layers.Bidirectional(layers.LSTM(SIZE_RLAYERS, return_sequences=True,
#                                                   kernel_regularizer=l2(L2R),
#                                                   recurrent_regularizer=l2(L2R),
#                                                   bias_regularizer=l2(L2R),
#                                                   dropout=DROPOUT,
#                                                   recurrent_dropout=RDROPOUT),
#                                       input_shape=[convoutput._keras_shape[2],convoutput._keras_shape[3]]))  # (32, 64) (frequency, #kernles)
#     rnnModel.add(layers.Bidirectional(layers.LSTM(SIZE_RLAYERS, return_sequences=True,
#                                                   kernel_regularizer=l2(L2R),
#                                                   recurrent_regularizer=l2(L2R),
#                                                   bias_regularizer=l2(L2R),
#                                                   dropout=DROPOUT,
#                                                   recurrent_dropout=RDROPOUT)))
#
#     rnnModel.add(layers.TimeDistributed(layers.Dense(4,  # 32*4~127
#                                                      activation='sigmoid',
#                                                      kernel_regularizer=l2(L2R),
#                                                      bias_regularizer=l2(L2R)),
#                                         name='mask_o'))
#
#     x = layers.TimeDistributed(rnnModel, name='RNN_f')(convoutput)
#
#     x_f = layers.Reshape((x._keras_shape[1], np.prod(x._keras_shape[2::])))(x)
#
#
#
#     SIZE_RLAYERS2 = 128
#     for i in range(2): # two stacked BiLSTM
#         x_f = layers.Bidirectional(layers.LSTM(SIZE_RLAYERS2, return_sequences=True,
#                                              kernel_regularizer=l2(L2R),
#                                              recurrent_regularizer=l2(L2R),
#                                              bias_regularizer=l2(L2R),
#                                              dropout=DROPOUT,
#                                              recurrent_dropout=RDROPOUT))(x_f)
#     mask_o = layers.TimeDistributed(layers.Dense(out_shape[-1],
#                                      activation='sigmoid',
#                                      kernel_regularizer=l2(L2R),
#                                      bias_regularizer=l2(L2R)),
#                                      name='mask_o')(x_f)
#
#     train_model = models.Model(inputs=[inp], outputs=[mask_o])
#
#     return train_model

# def GenerateBLSTMTFOld():
#     # The same pre-processing is appled to different DNN structures
#     [inp_shape, out_shape, inp, convoutput] = preprocess()
#
#     # The RNN model that being applied to each frame
#     SIZE_RLAYERS = 32
#     # Regularization parameters
#     DROPOUT = 0.25  # Feed forward dropout
#     RDROPOUT = 0.1  # Recurrent dropout
#     L2R = 1e-6  # L2 regularization factor
#
#     rnnModel = models.Sequential(name='BiLSTM_f')
#     rnnModel.add(layers.Bidirectional(layers.LSTM(SIZE_RLAYERS, return_sequences=True,
#                                                   kernel_regularizer=l2(L2R),
#                                                   recurrent_regularizer=l2(L2R),
#                                                   bias_regularizer=l2(L2R),
#                                                   dropout=DROPOUT,
#                                                   recurrent_dropout=RDROPOUT),
#                                       input_shape=[convoutput._keras_shape[2],convoutput._keras_shape[3]]))  # (32, 64) (frequency, #kernles)
#     # rnnModel.add(layers.Bidirectional(layers.LSTM(SIZE_RLAYERS, return_sequences=True,
#     #                                               kernel_regularizer=l2(L2R),
#     #                                               recurrent_regularizer=l2(L2R),
#     #                                               bias_regularizer=l2(L2R),
#     #                                               dropout=DROPOUT,
#     #                                               recurrent_dropout=RDROPOUT)))
#
#     # rnnModel.add(layers.TimeDistributed(layers.Dense(4,  # 32*4~127
#     #                                                  activation='relu',
#     #                                                  kernel_regularizer=l2(L2R),
#     #                                                  bias_regularizer=l2(L2R)),
#     #                                     name='mask_o'))
#
#     x = layers.TimeDistributed(rnnModel, name='RNN_f')(convoutput)
#
#     x_f = layers.Reshape((x._keras_shape[1], np.prod(x._keras_shape[2::])))(x)
#
#     SIZE_RLAYERS2 = 256
#     DROPOUT2 = 0.5  # Feed forward dropout
#     RDROPOUT2 = 0.2  # Recurrent dropout
#     x_f = layers.Bidirectional(layers.LSTM(SIZE_RLAYERS2, return_sequences=True,
#                                            kernel_regularizer=l2(L2R),
#                                            recurrent_regularizer=l2(L2R),
#                                            bias_regularizer=l2(L2R),
#                                            dropout=DROPOUT2,
#                                            recurrent_dropout=RDROPOUT2))(x_f)
#     mask_o = layers.TimeDistributed(layers.Dense(out_shape[-1],
#                                      activation='sigmoid',
#                                      kernel_regularizer=l2(L2R),
#                                      bias_regularizer=l2(L2R)),
#                                      name='mask_o')(x_f)
#
#     train_model = models.Model(inputs=[inp], outputs=[mask_o])
#
#     return train_model

def GenerateBLSTMTF3():
    # The same pre-processing is appled to different DNN structures
    [inp_shape, out_shape, inp, convoutput] = preprocess()

    # RNN along the frequency bins
    SIZE_RLAYERS = 128
    # Regularization parameters
    DROPOUT = 0.5  # Feed forward dropout
    RDROPOUT = 0.2  # Recurrent dropout
    L2R = 1e-6  # L2 regularization factor

    rnnModel = models.Sequential(name='BiLSTM_f')
    rnnModel.add(layers.Bidirectional(layers.LSTM(SIZE_RLAYERS, return_sequences=True,
                                                  kernel_regularizer=l2(L2R),
                                                  recurrent_regularizer=l2(L2R),
                                                  bias_regularizer=l2(L2R),
                                                  dropout=DROPOUT,
                                                  recurrent_dropout=RDROPOUT),
                                      input_shape=[convoutput._keras_shape[2],
                                                   convoutput._keras_shape[3]]))  # (32, 64) (frequency, #kernles)


    x = layers.TimeDistributed(rnnModel, name='RNN_f')(convoutput)
    tmp = np.prod(x._keras_shape[2::])/128
    y_f = layers.Reshape((x._keras_shape[1], 128, tmp))(x)  # (100, 128)

    # RNN along the frames
    convoutputR = layers.Reshape((convoutput._keras_shape[1], np.prod(convoutput._keras_shape[2::])))(convoutput)  # (100, 128)


    SIZE_RLAYERS2 = 256

    rnnModel2 = models.Sequential(name='BiLSTM_t')
    rnnModel2.add(layers.Bidirectional(layers.LSTM(SIZE_RLAYERS2, return_sequences=True,
                                                  kernel_regularizer=l2(L2R),
                                                  recurrent_regularizer=l2(L2R),
                                                  bias_regularizer=l2(L2R),
                                                  dropout=DROPOUT,
                                                  recurrent_dropout=RDROPOUT),
                                      input_shape=[convoutputR._keras_shape[1],
                                                   convoutputR._keras_shape[2]]))

    x2 = rnnModel2(convoutputR)
    tmp2 = SIZE_RLAYERS2*2/128
    y_t = layers.Reshape((100, 128, tmp2))(x2)

    y = layers.concatenate([y_f,y_t],axis=-1)

    # post processing with more conv layers
    zz1 = layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same')(y)
    z1 = layers.BatchNormalization()(zz1)
    z1 = layers.Activation('relu')(z1)

    zz2 = layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same')(z1)
    z2 = layers.BatchNormalization()(zz2)
    z2 = layers.Activation('relu')(z2)

    zz3 = layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same')(z2)
    z3sum = layers.Add()([zz1, zz3])
    z3 = layers.BatchNormalization()(z3sum)
    z3 = layers.Activation('relu')(z3)

    y_o = layers.Conv2D(filters=1, kernel_size=(5, 5), padding='same')(z3)
    y_o = layers.Activation('sigmoid')(y_o)

    def easyreshapesplit(x):
        #y = K.reshape(x, shape=[-1, x._keras_shape[1], np.prod(x._keras_shape[2::])])#(100, 128) (time, #freq)
        y = K.squeeze(x,axis=-1)#(100, 128) (time, #freq)
        output, _ = tf.split(y, [127, 1], axis=2)
        return output

    mask_o = layers.Lambda(easyreshapesplit, name='output')(y_o)






    train_model = models.Model(inputs=[inp], outputs=[mask_o])

    return train_model

def GenerateBLSTMTF():
    # The same pre-processing is appled to different DNN structures
    [inp_shape, out_shape, inp, convoutput] = preprocess()

    # RNN along the frequency bins
    SIZE_RLAYERS = 32
    # Regularization parameters
    DROPOUT = 0.2  # Feed forward dropout
    RDROPOUT = 0.1  # Recurrent dropout
    L2R = 1e-6  # L2 regularization factor

    rnnModel = models.Sequential(name='BiLSTM_f')
    rnnModel.add(layers.Bidirectional(layers.LSTM(SIZE_RLAYERS, return_sequences=True,
                                                  kernel_regularizer=l2(L2R),
                                                  recurrent_regularizer=l2(L2R),
                                                  bias_regularizer=l2(L2R),
                                                  dropout=DROPOUT,
                                                  recurrent_dropout=RDROPOUT),
                                      input_shape=[convoutput._keras_shape[2],
                                                   convoutput._keras_shape[3]]))  # (32, 64) (frequency, #kernles)
    rnnModel.add(layers.Bidirectional(layers.LSTM(SIZE_RLAYERS, return_sequences=True,
                                                  kernel_regularizer=l2(L2R),
                                                  recurrent_regularizer=l2(L2R),
                                                  bias_regularizer=l2(L2R),
                                                  dropout=DROPOUT,
                                                  recurrent_dropout=RDROPOUT)))


    x = layers.TimeDistributed(rnnModel, name='RNN_f')(convoutput)
    tmp = np.prod(x._keras_shape[2::])/128
    y_f = layers.Reshape((x._keras_shape[1], 128, tmp))(x)  # (100, 128)

    # RNN along the frames
    convoutputR = layers.Reshape((convoutput._keras_shape[1], np.prod(convoutput._keras_shape[2::])))(convoutput)  # (100, 128)


    SIZE_RLAYERS2 = 256
    DROPOUT2 = 0.5  # Feed forward dropout
    RDROPOUT2 = 0.2  # Recurrent dropout

    rnnModel2 = models.Sequential(name='BiLSTM_t')
    rnnModel2.add(layers.Bidirectional(layers.LSTM(SIZE_RLAYERS2, return_sequences=True,
                                                  kernel_regularizer=l2(L2R),
                                                  recurrent_regularizer=l2(L2R),
                                                  bias_regularizer=l2(L2R),
                                                  dropout=DROPOUT2,
                                                  recurrent_dropout=RDROPOUT2),
                                      input_shape=[convoutputR._keras_shape[1],
                                                   convoutputR._keras_shape[2]]))
    rnnModel2.add(layers.Bidirectional(layers.LSTM(SIZE_RLAYERS2, return_sequences=True,
                                                   kernel_regularizer=l2(L2R),
                                                   recurrent_regularizer=l2(L2R),
                                                   bias_regularizer=l2(L2R),
                                                   dropout=DROPOUT2,
                                                   recurrent_dropout=RDROPOUT2)))

    x2 = rnnModel2(convoutputR)
    tmp2 = SIZE_RLAYERS2*2/128
    y_t = layers.Reshape((100, 128, tmp2))(x2)

    y = layers.concatenate([y_f,y_t],axis=-1)

    # post processing with more conv layers
    zz1 = layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same')(y)
    z1 = layers.BatchNormalization()(zz1)
    z1 = layers.Activation('relu')(z1)

    zz2 = layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same')(z1)
    z2 = layers.BatchNormalization()(zz2)
    z2 = layers.Activation('relu')(z2)

    zz3 = layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same')(z2)
    z3sum = layers.Add()([zz1, zz3])
    z3 = layers.BatchNormalization()(z3sum)
    z3 = layers.Activation('relu')(z3)

    y_o = layers.Conv2D(filters=1, kernel_size=(5, 5), padding='same')(z3)
    y_o = layers.Activation('sigmoid')(y_o)

    def easyreshapesplit(x):
        #y = K.reshape(x, shape=[-1, x._keras_shape[1], np.prod(x._keras_shape[2::])])#(100, 128) (time, #freq)
        y = K.squeeze(x,axis=-1)#(100, 128) (time, #freq)
        output, _ = tf.split(y, [127, 1], axis=2)
        return output

    mask_o = layers.Lambda(easyreshapesplit, name='output')(y_o)






    train_model = models.Model(inputs=[inp], outputs=[mask_o])

    return train_model



def GenerateBLSTMTimeDC():
    # The same pre-processing is appled to different DNN structures
    [inp_shape, out_shape, inp, convoutput] = preprocess()


    def easyreshape(x):
        xR = K.reshape(x, shape=[-1, 100, np.prod(convoutput._keras_shape[2::])])
        return xR

    convoutputR = layers.Lambda(easyreshape, name='reshape2')(convoutput)


    SIZE_RLAYERS = 256
    # Regularization parameters
    DROPOUT = 0.5  # Feed forward dropout
    RDROPOUT = 0.2  # Recurrent dropout
    L2R = 1e-6  # L2 regularization factor

    # #dimension reduction in each frame
    # simpleModel = models.Sequential(name='dense_layer')
    # # simpleModel.add(layers.Dropout(0.5, input_shape=(convoutputR._keras_shape[-1],)))
    # # simpleModel.add(layers.Dense(256, activation='relu', kernel_regularizer=l2(L2R), bias_regularizer=l2(L2R)))
    # simpleModel.add(layers.Dense(256, activation='relu', kernel_regularizer=l2(L2R), bias_regularizer=l2(L2R),input_shape=(convoutputR._keras_shape[-1],)))
    # simpleModel.add(layers.BatchNormalization())
    #
    # x = layers.TimeDistributed(simpleModel,name='Dense')(convoutputR)

    x = convoutputR

    for i in range(2): # two stacked BiLSTM
        x = layers.Bidirectional(layers.LSTM(SIZE_RLAYERS, return_sequences=True,
                               kernel_regularizer=l2(L2R),
                               recurrent_regularizer=l2(L2R),
                               bias_regularizer=l2(L2R),
                               dropout=DROPOUT,
                               recurrent_dropout=RDROPOUT))(x)

    EMBEDDINGS_DIM = 40
    cluster_o = layers.TimeDistributed(layers.Dense(out_shape[-1]*EMBEDDINGS_DIM,
                                     activation='tanh',
                                     kernel_regularizer=l2(L2R),
                                     bias_regularizer=l2(L2R)),
                               name='cluster_o')(x)

    train_model = models.Model(inputs=[inp], outputs=[cluster_o])

    return train_model


def GenerateBLSTMFrequencyDC():
    # The same pre-processing is appled to different DNN structures
    [inp_shape, out_shape, inp, convoutput] = preprocess()

    # The RNN model that being applied to each frame
    SIZE_RLAYERS = 128
    # Regularization parameters
    DROPOUT = 0.5  # Feed forward dropout
    RDROPOUT = 0.2  # Recurrent dropout
    L2R = 1e-6  # L2 regularization factor

    rnnModel = models.Sequential(name='BiLSTM_f')
    rnnModel.add(layers.Bidirectional(layers.LSTM(SIZE_RLAYERS, return_sequences=True,
                                             kernel_regularizer=l2(L2R),
                                             recurrent_regularizer=l2(L2R),
                                             bias_regularizer=l2(L2R),
                                             dropout=DROPOUT,
                                             recurrent_dropout=RDROPOUT),
                                      input_shape=[convoutput._keras_shape[2], convoutput._keras_shape[3]]))#(32, 64) (frequency, #kernles)
    rnnModel.add(layers.Bidirectional(layers.LSTM(SIZE_RLAYERS, return_sequences=True,
                                                  kernel_regularizer=l2(L2R),
                                                  recurrent_regularizer=l2(L2R),
                                                  bias_regularizer=l2(L2R),
                                                  dropout=DROPOUT,
                                                  recurrent_dropout=RDROPOUT)))
    EMBEDDINGS_DIM = 40
    rnnModel.add(layers.TimeDistributed(layers.Dense(4*EMBEDDINGS_DIM, # 32*4~127
                                                 activation='sigmoid',
                                                 kernel_regularizer=l2(L2R),
                                                 bias_regularizer=l2(L2R)),
                                    name='mask_o'))

    x = layers.TimeDistributed(rnnModel, name='RNN_f')(convoutput)

    def easyreshapesplit(x):
        #y = K.reshape(x, shape=[-1, x._keras_shape[1], np.prod(x._keras_shape[2::])])#(100, 128) (time, #freq)
        y = layers.Reshape((x._keras_shape[1], np.prod(x._keras_shape[2::])))(x)#(100, 128) (time, #freq)
        output, _ = tf.split(y, [127*EMBEDDINGS_DIM, 1*EMBEDDINGS_DIM], axis=2)
        return output

    y = layers.Lambda(easyreshapesplit, name='reshape3')(x)

    train_model = models.Model(inputs=[inp], outputs=[y])

    return train_model

def GenerateBLSTMTFDC():
    # The same pre-processing is appled to different DNN structures
    [inp_shape, out_shape, inp, convoutput] = preprocess()

    # The RNN model that being applied to each frame
    SIZE_RLAYERS = 32
    # Regularization parameters
    DROPOUT = 0.5  # Feed forward dropout
    RDROPOUT = 0.2  # Recurrent dropout
    L2R = 1e-6  # L2 regularization factor

    rnnModel = models.Sequential(name='BiLSTM_f')
    rnnModel.add(layers.Bidirectional(layers.LSTM(SIZE_RLAYERS, return_sequences=True,
                                                  kernel_regularizer=l2(L2R),
                                                  recurrent_regularizer=l2(L2R),
                                                  bias_regularizer=l2(L2R),
                                                  dropout=DROPOUT,
                                                  recurrent_dropout=RDROPOUT),
                                      input_shape=[convoutput._keras_shape[2],convoutput._keras_shape[3]]))  # (32, 64) (frequency, #kernles)


    x = layers.TimeDistributed(rnnModel, name='RNN_f')(convoutput)

    x_f = layers.Reshape((x._keras_shape[1], np.prod(x._keras_shape[2::])))(x)



    SIZE_RLAYERS2 = 256
    x_f = layers.Bidirectional(layers.LSTM(SIZE_RLAYERS2, return_sequences=True,
                                         kernel_regularizer=l2(L2R),
                                         recurrent_regularizer=l2(L2R),
                                         bias_regularizer=l2(L2R),
                                         dropout=DROPOUT,
                                         recurrent_dropout=RDROPOUT))(x_f)
    EMBEDDINGS_DIM = 40
    cluster_o = layers.TimeDistributed(layers.Dense(out_shape[-1] * EMBEDDINGS_DIM,
                                     activation='sigmoid',
                                     kernel_regularizer=l2(L2R),
                                     bias_regularizer=l2(L2R)),
                                     name='cluster_o')(x_f)


    train_model = models.Model(inputs=[inp], outputs=[cluster_o])

    return train_model



if __name__=="__main__":
    train_model = GenerateBLSTMTF()
    print(train_model.summary())
    from keras.utils import plot_model
    plot_model(train_model, to_file='Model.png')






# def GenerateBLSTMFrequencyOld():
#     [inp_shape, out_shape, inp, convoutput] = preprocess()
#
#     # aa = np.random.randn(2,100,10)
#     # bb = tf.convert_to_tensor(aa)
#     # cc = K.reshape(bb,[200,10])
#
#     def easyreshape(convoutput):
#         convoutputR = K.reshape(convoutput, shape=[-1, convoutput._keras_shape[2], convoutput._keras_shape[3] ])#(32, 64) (frequency, #kernles)
#         return convoutputR
#
#     convoutputR = layers.Lambda(easyreshape, name='reshape2')(convoutput)
#
#     x = convoutputR
#
#
#     SIZE_RLAYERS = 128
#     # Regularization parameters
#     DROPOUT = 0.5  # Feed forward dropout
#     RDROPOUT = 0.2  # Recurrent dropout
#     L2R = 1e-6  # L2 regularization factor
#
#     for i in range(2): # two stacked BiLSTM
#         x = layers.Bidirectional(layers.LSTM(SIZE_RLAYERS, return_sequences=True,
#                                              kernel_regularizer=l2(L2R),
#                                              recurrent_regularizer=l2(L2R),
#                                              bias_regularizer=l2(L2R),
#                                              dropout=DROPOUT,
#                                              recurrent_dropout=RDROPOUT))(x)
#     mask_o = layers.TimeDistributed(layers.Dense(4, # 32*4~127
#                                                  activation='sigmoid',
#                                                  kernel_regularizer=l2(L2R),
#                                                  bias_regularizer=l2(L2R)),
#                                     name='mask_o')(x)
#
#     def easyreshapesplit(x):
#         y = K.reshape(x, shape=[-1, 100, np.prod(mask_o._keras_shape[1::])])#(100, 128) (time, #freq)
#         output, _ = tf.split(y, [127, 1], axis=2)
#         return output
#
#     y = layers.Lambda(easyreshapesplit, name='reshape3')(mask_o)
#
#
#     train_model = models.Model(inputs=[inp], outputs=[y])
#
#     return train_model
#
# def GenerateBLSTMTFOld():
#     [inp_shape, out_shape, inp, convoutput] = preprocess()
#
#     def easyreshape(convoutput):
#         convoutputR = K.reshape(convoutput, shape=[-1, convoutput._keras_shape[2], convoutput._keras_shape[3]])  # (32, 64) (frequency, #kernles)
#         return convoutputR
#
#     convoutputR = layers.Lambda(easyreshape, name='reshape2')(convoutput)
#
#     x = convoutputR
#
#
#     SIZE_RLAYERS = 32
#     # Regularization parameters
#     DROPOUT = 0.5  # Feed forward dropout
#     RDROPOUT = 0.2  # Recurrent dropout
#     L2R = 1e-6  # L2 regularization factor
#
#     for i in range(1): # two stacked BiLSTM
#         x = layers.Bidirectional(layers.LSTM(SIZE_RLAYERS, return_sequences=True,
#                                              kernel_regularizer=l2(L2R),
#                                              recurrent_regularizer=l2(L2R),
#                                              bias_regularizer=l2(L2R),
#                                              dropout=DROPOUT,
#                                              recurrent_dropout=RDROPOUT))(x)
#
#     def easyreshape2(x):
#         y = K.reshape(x, shape=[-1, 100, np.prod(x._keras_shape[2::])])#(100, 2048) (time, #freq features)
#         return y
#
#     x_f = layers.Lambda(easyreshape2, name='reshape3')(x)
#
#
#     # x_f = layers.Reshape([-1, 100, (2*SIZE_RLAYERS)*convoutput._keras_shape[2]])(x)
#
#     SIZE_RLAYERS2 = 128
#     for i in range(2): # two stacked BiLSTM
#         x = layers.Bidirectional(layers.LSTM(SIZE_RLAYERS2, return_sequences=True,
#                                              kernel_regularizer=l2(L2R),
#                                              recurrent_regularizer=l2(L2R),
#                                              bias_regularizer=l2(L2R),
#                                              dropout=DROPOUT,
#                                              recurrent_dropout=RDROPOUT))(x_f)
#     mask_o = layers.TimeDistributed(layers.Dense(out_shape[-1],
#                                      activation='sigmoid',
#                                      kernel_regularizer=l2(L2R),
#                                      bias_regularizer=l2(L2R)),
#                                      name='mask_o')(x_f)
#
#     train_model = models.Model(inputs=[inp], outputs=[mask_o])
#
#     return train_model
#
#

# def GenerateBLSTMTime():
#     inp_shape = (100, 127 * 3)
#     out_shape = (100, 127)
#
#     inp = layers.Input(shape=inp_shape, name='input')
#
#     x = inp
#
#
#     SIZE_RLAYERS = 256
#     # Regularization parameters
#     DROPOUT = 0.5  # Feed forward dropout
#     RDROPOUT = 0.2  # Recurrent dropout
#     L2R = 1e-6  # L2 regularization factor
#
#     for i in range(2): # two stacked BiLSTM
#         x = layers.Bidirectional(layers.LSTM(SIZE_RLAYERS, return_sequences=True,
#                                W_regularizer=l2(L2R),
#                                U_regularizer=l2(L2R),
#                                b_regularizer=l2(L2R),
#                                dropout_W=DROPOUT,
#                                dropout_U=RDROPOUT))(x)
#     mask_o = layers.TimeDistributed(layers.Dense(out_shape[-1],
#                                      activation='sigmoid',
#                                      W_regularizer=l2(L2R),
#                                      b_regularizer=l2(L2R)),
#                                name='mask_o')(x)
#
#     train_model = models.Model(input=[inp], output=[mask_o])
#
#     return train_model
#
# def GenerateBLSTMFrequency():
#     inp_shape = (100, 127 * 3)
#     out_shape = (100, 127)
#
#     inp = layers.Input(shape=inp_shape, name='input')
#
#
#     def easyreshape(input):
#         # inp_shape = (100, 127 * 3)
#         inputR = K.stack(tf.split(input, 3, axis=2), axis=-1)  # (100, 127, 3)
#         inputRR = K.reshape(inputR, shape=[-1, inputR._shape[2], inputR._shape[3] ])#(127, 3) (frequency, #kernles)
#         return inputRR
#
#     x = layers.Lambda(easyreshape, name='reshape')(inp)
#
#     SIZE_RLAYERS = 32
#     # Regularization parameters
#     DROPOUT = 0.5  # Feed forward dropout
#     RDROPOUT = 0.2  # Recurrent dropout
#     L2R = 1e-6  # L2 regularization factor
#
#     for i in range(2): # two stacked BiLSTM
#         x = layers.Bidirectional(layers.LSTM(SIZE_RLAYERS, return_sequences=True,
#                                W_regularizer=l2(L2R),
#                                U_regularizer=l2(L2R),
#                                b_regularizer=l2(L2R),
#                                dropout_W=DROPOUT,
#                                dropout_U=RDROPOUT))(x)
#     mask_o = layers.TimeDistributed(layers.Dense(1, # 32*4~127
#                                      activation='sigmoid',
#                                      W_regularizer=l2(L2R),
#                                      b_regularizer=l2(L2R)),
#                                name='mask_o')(x)
#
#     def easyreshape(x):
#         y = K.reshape(x, shape=[-1, 100, 127])
#         return y
#
#     y = layers.Lambda(easyreshape, name='reshape2')(mask_o)
#
#
#
#
#     train_model = models.Model(input=[inp], output=[y])
#
#     return train_model
#
# def GenerateBLSTMTF():
#     inp_shape = (100, 127 * 3)
#     out_shape = (100, 127)
#
#     inp = layers.Input(shape=inp_shape, name='input')
#
#     def easyreshape(input):
#         # inp_shape = (100, 127 * 3)
#         inputR = K.stack(tf.split(input, 3, axis=2), axis=-1)  # (100, 127, 3)
#         inputRR = K.reshape(inputR, shape=[-1, inputR._shape[2], inputR._shape[3]])  # (127, 3) (frequency, #kernles)
#         return inputRR
#
#     x = layers.Lambda(easyreshape, name='reshape')(inp)
#
#
#     SIZE_RLAYERS = 32
#     # Regularization parameters
#     DROPOUT = 0.5  # Feed forward dropout
#     RDROPOUT = 0.2  # Recurrent dropout
#     L2R = 1e-6  # L2 regularization factor
#
#     for i in range(1): # two stacked BiLSTM
#         x = layers.Bidirectional(layers.LSTM(SIZE_RLAYERS, return_sequences=True,
#                                W_regularizer=l2(L2R),
#                                U_regularizer=l2(L2R),
#                                b_regularizer=l2(L2R),
#                                dropout_W=DROPOUT,
#                                dropout_U=RDROPOUT))(x)
#     x = layers.TimeDistributed(layers.Dense(4,
#                                      activation='sigmoid',
#                                      W_regularizer=l2(L2R),
#                                      b_regularizer=l2(L2R)))(x)
#
#     def easyreshape2(x):
#         y = K.reshape(x, shape=[-1, 100, x._keras_shape[1]*x._keras_shape[2]])#(100, 2048) (time, #freq features)
#         return y
#
#     x_f = layers.Lambda(easyreshape2, name='reshape2')(x)
#
#
#     # x_f = layers.Reshape([-1, 100, (2*SIZE_RLAYERS)*convoutput._keras_shape[2]])(x)
#
#     SIZE_RLAYERS2 = 256
#     for i in range(2): # two stacked BiLSTM
#         x_f = layers.Bidirectional(layers.LSTM(SIZE_RLAYERS2, return_sequences=True,
#                                W_regularizer=l2(L2R),
#                                U_regularizer=l2(L2R),
#                                b_regularizer=l2(L2R),
#                                dropout_W=DROPOUT,
#                                dropout_U=RDROPOUT))(x_f)
#     mask_o = layers.TimeDistributed(layers.Dense(out_shape[-1],
#                                      activation='sigmoid',
#                                      W_regularizer=l2(L2R),
#                                      b_regularizer=l2(L2R)),
#                                      name='mask_o')(x_f)
#
#     train_model = models.Model(input=[inp], output=[mask_o])
#
#     return train_model
#
# def GenerateBLSTMFrequencyNew():
#     inp_shape = (100, 127 * 3)
#     out_shape = (100, 127)
#
#     convModel = models.Sequential(name='conv_layer')
#     convModel.add(layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 2), padding='valid', input_shape=(5, 131, 3), name='conv1'))
#     convModel.add(layers.BatchNormalization())
#     convModel.add(layers.LeakyReLU())
#
#     RNNModel = models.Sequential(name='lstm_layer')
#     SIZE_RLAYERS = 128
#     # Regularization parameters
#     DROPOUT = 0.5  # Feed forward dropout
#     RDROPOUT = 0.2  # Recurrent dropout
#     L2R = 1e-6  # L2 regularization factor
#     # two stacked BiLSTM
#     RNNModel.add(layers.Bidirectional(layers.LSTM(SIZE_RLAYERS, return_sequences=True,
#                                                   W_regularizer=l2(L2R),
#                                                   U_regularizer=l2(L2R),
#                                                   b_regularizer=l2(L2R),
#                                                   dropout_W=DROPOUT,
#                                                   dropout_U=RDROPOUT),
#                                       input_shape=(64,32)
#                                       )
#     )
#
#     RNNModel.add(layers.Bidirectional(layers.LSTM(SIZE_RLAYERS, return_sequences=True,
#                                                   W_regularizer=l2(L2R),
#                                                   U_regularizer=l2(L2R),
#                                                   b_regularizer=l2(L2R),
#                                                   dropout_W=DROPOUT,
#                                                   dropout_U=RDROPOUT)
#                                       )
#     )
#
#     RNNModel.add(layers.TimeDistributed(layers.Dense(10,  # 64*10~127*5 (64, 256)
#                                                  activation='sigmoid',
#                                                  W_regularizer=l2(L2R),
#                                                  b_regularizer=l2(L2R)),
#                                     name='mask_o')
#
#     )
#
#
#
#     def blockProcess(input):
#         # inp_shape = (5, 127 * 3)
#         def reshapestack(input):
#             inputR = K.stack(tf.split(input, 3, axis=2), axis=-1)  # (5, 127, 3)
#             return inputR
#
#         inputR = layers.Lambda(reshapestack)(input)
#         # inputR = K.stack(tf.split(input, 3, axis=2), axis=-1)  # (5, 127, 3)
#         inputR0 = layers.ZeroPadding2D(padding=(0, 2), data_format=None)(inputR)  # (5, 127+4, 3)
#
#         # Layer 1: A conventional Conv2D layer with batch noarmalisation and max pooling
#         convOutput = convModel(inputR0) #(1, 64, 32)
#         convOutputR = layers.Reshape(target_shape=(64, 32))(convOutput) #(64,32)
#
#         # apply frequency domain BiLSTM
#         output = RNNModel(convOutputR) #(64, 10)
#         outputR = layers.Reshape(target_shape=(128, 5))(output) #(128,5)
#         def Ktransposesplit(input):
#             outputRT = tf.transpose(outputR,[0, 2, 1])
#             outputFinal, _ = tf.split(outputRT, [127, 1], axis=2)  # (5, 127)
#             return outputFinal
#         outputFinal = layers.Lambda(Ktransposesplit)(outputR)
#
#         return outputFinal
#
#
#     inp = layers.Input(shape=inp_shape, name='input')
#     # y = layers.Lambda(myprocess, name='myProcess')(inp)  # (64, 32) (time, frequency, #kernles)
#
#     def split(input):
#         # Every 5 frames are processed
#         # inp_shape = (100, 127 * 3)
#         inputList = tf.split(input,20,axis=1) # each element (5, 127 * 3)
#         return inputList
#     inputList = layers.Lambda(split, name='inputsplit')(inp)  # (64, 32) (time, frequency, #kernles)
#
#     yList = [blockProcess(x) for x in inputList]
#
#     y = layers.Concatenate(axis=1)(yList)
#
#     train_model = models.Model(input=[inp], output=[y])
#
#     return train_model












