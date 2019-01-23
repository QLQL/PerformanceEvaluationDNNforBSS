# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:09:46 2016

@author: valterf

Main code with examples for the most important function calls. None of this
will work if you haven't prepared your train/valid/test file lists.
"""

from keras import backend as K
from keras import optimizers
from keras import callbacks
from keras.utils import plot_model



K.set_learning_phase(1) #set learning phase
K.set_image_data_format('channels_last')
# to check if there are gpu available
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

GPUFlag = True
ExistGPUs = get_available_gpus()
if len(ExistGPUs)==0:
    GPUFlag = False


# time mode 1, frequency mode 2, TF mode 3
modelMode = 3



from DataGenerator import dataGenBig
dg = dataGenBig()
dg.TrainDataParamsInit()
# [aa,bb] = dg.TrainDataGenerator()# change yield to return to debug the generator



# Load the model
#####################################################
##################Direct Regression #################
#####################################################
from Others import my_loss as customLoss
mode = 2

if modelMode == 1:
    from GenerateModels import GenerateBLSTMTime as GenerateBLSTM

    tag = 'TimeModel'

elif modelMode == 2:
    from GenerateModels import GenerateBLSTMFrequency as GenerateBLSTM

    tag = 'FrequencyModel'

elif modelMode == 3:
    from GenerateModels import GenerateBLSTMTF as GenerateBLSTM

    tag = 'TFModel'

print(tag)



# # Load the model
# #####################################################
# ##################Deep Clustering ###################
# #####################################################
# from Others import affinitykmeans as customLoss
# mode = 3
#
# if modelMode == 1:
#     from GenerateModels import GenerateBLSTMTimeDC as GenerateBLSTM
#
#     tag = 'TimeModelDC'
#
# elif modelMode == 2:
#     from GenerateModels import GenerateBLSTMFrequencyDC as GenerateBLSTM
#
#     tag = 'FrequencyModelDC'
#
# elif modelMode == 3:
#     from GenerateModels import GenerateBLSTMTFDC as GenerateBLSTM
#
#     tag = 'TFModelDC'





# Load the RNN model
train_model = GenerateBLSTM()
# print(train_model.summary())
# plot_model(train_model, to_file='FrequencyModel.png')



# setting the hyper parameters
import os
import argparse

# setting the hyper parameters
parser = argparse.ArgumentParser(description="RNNforBSS")
parser.add_argument('--epochs', default=1000, type=int)
parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
parser.add_argument('--save_dir', default='./result/{}'.format(tag))
parser.add_argument('--is_training', default=1, type=int)
parser.add_argument('-w', '--weights', default=None, help="The path of the saved weights. Should be specified when testing")
parser.add_argument('--lr', default=0.001, type=float, help="Initial learning rate")
parser.add_argument('--lr_decay', default=0.95, type=float, help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
args = parser.parse_args()
print(args)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)


# train_model.compile(loss={'kmeans_o': affinitykmeans}, optimizer=optimizers.Nadam(clipnorm=CLIPNORM))
# compile the model
train_model.compile(optimizer=optimizers.Adam(lr=args.lr), loss=[customLoss], metrics=[customLoss])

log = callbacks.CSVLogger(args.save_dir + '/log.csv')
tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                           batch_size=dg.BATCH_SIZE_Train, histogram_freq=args.debug)
checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_loss',
                                       save_best_only=True, save_weights_only=True, verbose=1, period=10)
lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

train_model.fit_generator(generator=dg.TrainDataGenerator(mode),
                    steps_per_epoch=int(dg.trainNum / dg.BATCH_SIZE_Train),
                    epochs=args.epochs,
                    validation_data=dg.ValidDataGenerator(mode),
                    validation_steps=min([80,int(dg.validNum / dg.BATCH_SIZE_Valid)]),
                    callbacks=[log, tb, checkpoint, lr_decay])
# End: Training with data augmentation -----------------------------------------------------------------------#

train_model.save_weights(args.save_dir + '/trained_model.h5')
print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)


# If you wish to test source separation, generate a mixed 'mixed.wav'
# file and test with the following line
# separate_sources('mixed.wav', model, 2, 'out')


