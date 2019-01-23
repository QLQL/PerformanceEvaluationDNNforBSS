import numpy as np
from keras import layers, models, optimizers, initializers
from keras import backend as K
import tensorflow as tf
from keras import callbacks
from keras.utils import plot_model
import scipy.io as sio
from Others import *


K.set_image_data_format('channels_last')


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())




#########################################
# Test the capsule Network
from GenerateModels import GenerateBSSBLSTMNet
modelDirectory = './result09Mar/BLSTMNet'
model = GenerateBSSBLSTMNet()



# The data to be separated
folderName = '/SomeDirectoryToChange/DLTestData07Mar/Matched/MM/-30'
filename = 'MM_Azi-30_5.mat'
mat = sio.loadmat(folderName + '/' + filename)

# The results to be saved
params_dir = '/SomeDirectoryToChange/Params/myNormParamsMixture.mat'
fullSaveDir = '/user/cvsspstf/ql0002/Desktop/'
save_mix_name = 'mix.wav'
save_est_name = 'est.wav'
save_gth_name = 'gth.wav'



recoverObj = RecoverData(params_dir, 0.25, 8000)



# The separation model
model.load_weights(modelDirectory + '/' + 'trained_model.h5')
print(model.summary())
# plot_model(train_model, to_file='CapsNetModel.png')



# from DataGenerator import dataGenBig
# dg = dataGenBig()
# (mixin, groundth, _) =dg.processOneBatchFromMat(mat, filename, batchFlag=False)

mixin = extractInputFeature(mat, filename)
groundth = mat['s1LogPower'][1:-1,5:-5].transpose()
# normalisation of the groundtruth signal
# for k in range(groundth.shape[0]):
#     groundth[k] -= recoverObj.sig_mean
#     groundth[k] /= recoverObj.sig_std
groundth -= recoverObj.sig_mean
groundth /= recoverObj.sig_std

#


# apply the source separation to get the normalised log spectrum
output_musk = model.predict(mixin)
# apply the mask value to the output_LP
mixLP = mixin[:,:,5,0].squeeze()
output = 2*np.log(output_musk) + mixLP
currentThreshold = np.max(output) - 25 # lower than 25 dB
output[output < currentThreshold] = currentThreshold

mixAngle_L = mat['mixAngle_L'][1:-1,5:-5].transpose()

recoverObj.recover_from_spectrum(mixin[:,:,5,0].squeeze(), mixAngle_L,
                                 fullSaveDir + save_mix_name)
recoverObj.recover_from_spectrum(output, mixAngle_L,
                                 fullSaveDir + save_est_name)
recoverObj.recover_from_spectrum(groundth, mixAngle_L,
                                 fullSaveDir + save_gth_name)

