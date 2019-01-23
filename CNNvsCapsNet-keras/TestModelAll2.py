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
modelModeArray = [5,6] # oracle results with IBM
MatchedFlag = False
if MatchedFlag:
    MatchedStr='Matched/'
    Azimuth_array = [-90, -60, -30, 30, 60, 90]
else:
    MatchedStr = 'Unmatched/'
    Azimuth_array = [-135, -110, 110, 135]


# The results to be saved
saveDirectoryTop = '/SomeDirectoryToChange/DLResultsConv/'
# saveDirectoryTop = '/SomeDirectoryToChange/DLResultsAsilomar/DLResultsConv04Apr/'# The data to be separated
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
        from GenerateModels import GenerateBSSFullNet as GenerateModel

        tag = 'MLPModel'
        saveflag = 'MLP'

    elif modelMode == 2:
        from GenerateModels import GenerateBSSCNNNet as GenerateModel

        tag = 'CNNModel'
        saveflag = 'CNN'

    elif modelMode == 3:
        from GenerateModels import GenerateBSSCapsNet as GenerateModel

        tag = 'CapsNetModel'
        saveflag = 'Caps'

    elif modelMode == 4:

        tag = 'OracleIBM'
        saveflag = 'Oracle'

    elif modelMode == 5:

        tag = 'OracleIRM'
        saveflag = 'IRM'

    elif modelMode == 6:

        tag = 'OracleHybrid'
        saveflag = 'Hybrid'

    print(tag)


    if modelMode<4:
        #########################################
        modelDirectory = './result04April/'+tag
        model = GenerateModel()

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

                mixin = extractInputFeature(mat, fileName)

                if modelMode==4: # ideal binary mask
                    s1LogPower = mat['s1LogPower']
                    s2LogPower = mat['s2LogPower']
                    output_mask = ((s1LogPower[1:-1, 5:-5] > s2LogPower[1:-1,5:-5]).astype(float)).transpose()
                    output_mask[output_mask<0.001] = 0.001
                elif modelMode==5: # ideal ratio mask
                    s1LogPower = mat['s1LogPower']
                    s2LogPower = mat['s2LogPower']
                    s1 = np.sqrt(np.exp(s1LogPower[1:-1, 5:-5])).transpose()
                    s2 = np.sqrt(np.exp(s2LogPower[1:-1, 5:-5])).transpose()
                    output_mask = s1/(s1+s2)
                    output_mask[output_mask < 0.001] = 0.001
                elif modelMode==6:
                    s1LogPower = mat['s1LogPower']
                    s2LogPower = mat['s2LogPower']
                    s1 = np.sqrt(np.exp(s1LogPower[1:-1, 5:-5])).transpose()
                    s2 = np.sqrt(np.exp(s2LogPower[1:-1, 5:-5])).transpose()
                    IBMindex = s1 > s2
                    output_mask = IBMindex.astype(float)
                    output_mask2 = s1 / (s1 + s2)
                    output_mask[~IBMindex] = output_mask2[~IBMindex]

                else:
                    # apply the source separation to get the normalised log spectrum
                    output_mask = model.predict(mixin)
                # apply the mask value to the output_LP
                mixLP = mixin[:, :, 5, 0].squeeze()
                output = (2/sig_std) * np.log(output_mask) + mixLP
                currentThreshold = np.max(output) - 25  # lower than 25 dB
                output[output < currentThreshold] = currentThreshold

                mixAngle_L = mat['mixAngle_L'][1:-1, 5:-5].transpose()

                if modelMode == 1:
                    save_mix_name = fileName + '_mix.wav'
                    recoverObj.recover_from_spectrum(mixLP.squeeze(), mixAngle_L,
                                                     fullSaveDir_azi + save_mix_name)

                save_est_name = fileName + '_est_{}.wav'.format(saveflag)
                recoverObj.recover_from_spectrum(output, mixAngle_L,
                                                 fullSaveDir_azi + save_est_name)

