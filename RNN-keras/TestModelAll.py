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
modelModeArray = range(1,4)
MatchedFlag = True
if MatchedFlag:
    MatchedStr='Matched/'
    Azimuth_array = [-90, -60, -30, 30, 60, 90]
else:
    MatchedStr = 'Unmatched/'
    Azimuth_array = [-135, -110, 110, 135]


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
    #     saveflag = 'TDC'
    #
    # elif modelMode == 2:
    #     from GenerateModels import GenerateBLSTMFrequencyDC as GenerateBLSTM
    #
    #     tag = 'FrequencyModelDC'
    #     saveflag = 'FDC'
    #
    # elif modelMode == 3:
    #     from GenerateModels import GenerateBLSTMTFDC as GenerateBLSTM
    #
    #     tag = 'TFModelDC'
    #     saveflag = 'TFDC'




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
    for genderFlag in ['MM', 'MF', 'FF']:
        fullSaveDir = saveDirectoryTop + MatchedStr + '{}'.format(genderFlag)
        dataDir = folderName + MatchedStr + '{}/'.format(genderFlag)

        for azimuth in Azimuth_array:
            fullSaveDir_azi = fullSaveDir + '/{}/'.format(azimuth)
            if not os.path.exists(fullSaveDir_azi):
                os.makedirs(fullSaveDir_azi)

            dataDir_azi = dataDir + '/{}'.format(azimuth)

            for sequence in range(40): # In total 40 examples to separate in each scenario
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

                # plt.figure(101)
                # ax1 = plt.subplot(311)
                # im1 = ax1.pcolor(mix_spec.transpose())
                # plt.colorbar(im1)
                # ax2 = plt.subplot(312, sharex=ax1)
                # im2 = ax2.pcolor(mask_spec.transpose())
                # plt.colorbar(im2)
                # ax2 = plt.subplot(313, sharex=ax1)
                # im2 = ax2.pcolor(output_spec.transpose())
                # plt.colorbar(im2)
                #
                # plt.show()


                if modelMode==1:
                    save_mix_name = fileName + '_mix.wav'
                    recoverObj.recover_from_spectrum(mix_spec, mixAngle_L, fullSaveDir_azi + save_mix_name)

                save_est_name = fileName + '_est_{}.wav'.format(saveflag)
                recoverObj.recover_from_spectrum(output_spec, mixAngle_L, fullSaveDir_azi + save_est_name)

