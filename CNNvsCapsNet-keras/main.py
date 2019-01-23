import numpy as np
from keras import layers, models, optimizers, initializers
from keras import backend as K
import tensorflow as tf
from keras import callbacks
from keras.utils import to_categorical, plot_model
from Others import my_loss


K.set_learning_phase(1) #set learning phase
K.set_image_data_format('channels_last')
# to check if there are gpu available
from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

GPUFlag = True
ExistGPUs = get_available_gpus()
if len(ExistGPUs)==0:
    GPUFlag = False



# MLP mode 1, CNN mode 2, CapsNet mode 3
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
    from GenerateModels import GenerateBSSFullNet as GenerateModel

    tag = 'MLPModel'

elif modelMode == 2:
    from GenerateModels import GenerateBSSCNNNet as GenerateModel

    tag = 'CNNModel'

elif modelMode == 3:
    from GenerateModels import GenerateBSSCapsNet as GenerateModel

    tag = 'CapsNetModel'

print(tag)


# setting the hyper parameters
import os
import argparse

# setting the hyper parameters
parser = argparse.ArgumentParser(description="{} for BSS.".format(tag))
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('-r', '--routings', default=3, type=int, help="Number of iterations used in routing algorithm. should > 0")
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







# Load the RNN model
train_model = GenerateModel()
print(train_model.summary())
plot_model(train_model, to_file='{}.png'.format(tag))

log = callbacks.CSVLogger(args.save_dir + '/log.csv')
tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                           batch_size=dg.BATCH_SIZE_Train, histogram_freq=args.debug)
checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_loss',
                                       save_best_only=True, save_weights_only=True, verbose=1, period=10)
lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

# compile the model
train_model.compile(optimizer=optimizers.Adam(lr=args.lr), loss=[customLoss], metrics=[customLoss])




train_model.fit_generator(generator=dg.TrainDataGenerator(),
                    steps_per_epoch=int((dg.trainNum*dg.NeachSeq) / dg.BATCH_SIZE_Train),
                    epochs=args.epochs,
                    validation_data=dg.ValidDataGenerator(),
                    validation_steps=min([80,int((dg.validNum*dg.NeachSeq) / dg.BATCH_SIZE_Valid)]),
                    callbacks=[log, tb, checkpoint, lr_decay])
# End: Training with data augmentation -----------------------------------------------------------------------#

train_model.save_weights(args.save_dir + '/trained_model.h5')
print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

import matplotlib
matplotlib.use('Agg')
from utils import plot_log
plot_log(args.save_dir + '/log.csv', show=True)



# # # train or test
# # if args.weights is not None:  # init the model weights with provided one
# #     train_model.load_weights(args.weights)
# # if args.is_training:
# #     train(model=train_model, data=dg, args=args)
# # else:  # as long as weights are given, will run testing
# #     if args.weights is None:
# #         print('No weights are provided. Will test using random initialized weights.')
#
