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



modelModeArray = range(1,5)
MatchedFlag = True
if MatchedFlag:
    MatchedStr='Matched/'
    Azimuth_array = [60]
else:
    MatchedStr = 'Unmatched/'
    Azimuth_array = [135]



# The results to be saved
saveDirectoryTop = '/SomeDirectoryToChange/DLResultsConv/'# The data to be separated
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

                mixin = extractInputFeature(mat, fileName)

                if modelMode==4:
                    s1LogPower = mat['s1LogPower']
                    s2LogPower = mat['s2LogPower']
                    output_mask = ((s1LogPower[1:-1, 5:-5] > s2LogPower[1:-1,5:-5]).astype(float)).transpose()
                    output_mask[output_mask<0.001] = 0.001
                else:
                    # apply the source separation to get the normalised log spectrum
                    output_mask = model.predict(mixin)

                save_est_name = fileName + '_{}'.format(saveflag)

                sio.savemat('./{}.mat'.format(save_est_name), mdict={'output_mask': output_mask})



