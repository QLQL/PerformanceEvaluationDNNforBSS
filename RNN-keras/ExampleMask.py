import numpy as np
from keras import layers, models, optimizers, initializers
from keras import backend as K
import tensorflow as tf
from keras import callbacks
from keras.utils import plot_model
import scipy.io as sio
from Others import *

K.set_learning_phase(1) #set learning phase
K.set_image_data_format('channels_last')
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# modelModeArray = range(1,4)
modelModeArray = range(1,3)
MatchedFlag = True
if MatchedFlag:
    MatchedStr='Matched/'
    Azimuth_array = [60]
else:
    MatchedStr = 'Unmatched/'
    Azimuth_array = [135]


# The results to be saved
saveDirectoryTop = '/SomeDirectoryToChange/DLResultsBiLSTM/'
# The data to be separated
folderName = '/SomeDirectoryToChange/DLTestData07Mar/'
params_dir = '/SomeDirectoryToChange/Params/myNormParamsMixture.mat'
recoverObj = RecoverData(params_dir, 0.25, 8000)
mat2 = sio.loadmat(params_dir)
sig_std = np.asscalar(mat2['sig_std'])  # global std




for modelMode in modelModeArray:

    # Load the model
    #####################################################
    ##################Direct Regression #################
    #####################################################
    from Others import my_loss as customLoss
    mode = 2

    if modelMode == 1:
        from GenerateModels import GenerateBLSTMTime as GenerateBLSTM

        tag = 'TimeModel'
        saveflag = 'T'

    elif modelMode == 2:
        from GenerateModels import GenerateBLSTMFrequency as GenerateBLSTM

        tag = 'FrequencyModel'
        saveflag = 'F'

    elif modelMode == 3:
        from GenerateModels import GenerateBLSTMTF as GenerateBLSTM

        tag = 'TFModel'
        saveflag = 'TF'

    modelDirectory = './result01Apr/'+tag
    # Load the RNN model
    model = GenerateBLSTM()
    # print(model.summary())
    # plot_model(model, to_file='FrequencyModel.png')

    # The separation model
    model.load_weights(modelDirectory + '/' + 'trained_model.h5')
    print(model.summary())
    # plot_model(train_model, to_file='Model.png')


    import os
    for genderFlag in ['MF']:
        fullSaveDir = saveDirectoryTop + MatchedStr + '{}'.format(genderFlag)
        dataDir = folderName + MatchedStr + '{}/'.format(genderFlag)

        for azimuth in Azimuth_array:
            fullSaveDir_azi = fullSaveDir + '/{}/'.format(azimuth)
            if not os.path.exists(fullSaveDir_azi):
                os.makedirs(fullSaveDir_azi)

            dataDir_azi = dataDir + '/{}'.format(azimuth)

            for sequence in [20]: # In total 40 examples to separate in each scenario
                fileName = '{}_Azi{}_{}'.format(genderFlag, azimuth, sequence + 1)
                print(fileName)
                mat = sio.loadmat(dataDir_azi + '/' + fileName)

                (mixin, chooseIndex) = extractInputFeature(mat, fileName, mode)

                # apply the source separation to get the normalised log spectrum
                output_mask = model.predict(mixin)
                # apply the mask value to the output_LP
                mixLP = mixin[:, :, 0:127]
                output = (2/sig_std) * np.log(output_mask) + mixLP # normalised
                currentThreshold = np.max(output) - 25  # lower than 25 dB
                output[output < currentThreshold] = currentThreshold

                mixAngle_L = mat['mixAngle_L'][1:-1].transpose()

                output_spec = scipy.zeros_like(mixAngle_L)
                for n, i in enumerate(chooseIndex[:-1]):
                    output_spec[i:i + 100] = output[n].squeeze()
                # The last block of 100 frame
                tmp = chooseIndex[-1]
                output_spec[tmp:tmp + 100] = output[-1].squeeze()


                mix_spec = scipy.zeros_like(mixAngle_L)
                for n, i in enumerate(chooseIndex[:-1]):
                    mix_spec[i:i + 100] = mixin[n,:,0:127].squeeze()
                # The last block of 100 frame
                tmp = chooseIndex[-1]
                mix_spec[tmp:tmp + 100] = mixin[-1,:,0:127].squeeze()

                mask_spec = scipy.zeros_like(mixAngle_L)
                for n, i in enumerate(chooseIndex[:-1]):
                    mask_spec[i:i + 100] = output_mask[n].squeeze()
                # The last block of 100 frame
                tmp = chooseIndex[-1]
                mask_spec[tmp:tmp + 100] = output_mask[-1].squeeze()


                save_est_name = fileName + '_{}'.format(saveflag)
                sio.savemat('./{}.mat'.format(save_est_name), mdict={'output_mask': mask_spec})

