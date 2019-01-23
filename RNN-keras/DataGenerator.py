import numpy as np
import scipy.io as sio
import os
from random import shuffle
import random
import time
import re




class dataGenBig:
    def __init__(self, seedNum = 123456789, verbose = False):
        self.seedNum = seedNum
        self.verbose = verbose

        self.BATCH_SIZE_Train = 32 #32 # 128 #4096  # mini-batch size
        self.batchSeqN_Train = self.BATCH_SIZE_Train

        self.BATCH_SIZE_Valid = 32  # 32 # 128 #4096  # mini-batch size
        self.batchSeqN_Valid = self.BATCH_SIZE_Valid

        self.train_i = 0
        self.valid_i = 0


        # The mean and variance of the features to normalise the input and output feature vector
        mat2 = sio.loadmat('/SomeDirectoryToChange/Params/myNormParamsMixture.mat')
        # sig_std = mat2['sig_std_w']  # Dimx1
        # sig_mean = mat2['sig_mean_w']  # Dimx1
        # self.sig_std = sig_std.reshape(sig_std.size, )
        # self.sig_mean = sig_mean.reshape(sig_std.size, )
        self.sig_std = np.asscalar(mat2['sig_std'])  # global std
        self.sig_mean = np.asscalar(mat2['sig_mean'])  # global std

        # the IPD parameters extracted from clean signals
        # The mean and variance of the features to normalise the input and output feature vector
        mat1 = sio.loadmat('/SomeDirectoryToChange/Params/IPDParams.mat')
        IPD_mean = mat1['IPD_mean']
        self.IPD_mean = np.asarray(IPD_mean).transpose()
        IPD_var = mat1['IPD_var']
        # self.IPD_var = np.asarray(IPD_var).transpose()
        # self.IPD_var2 = -2*self.IPD_var
        # self.mlog2pisigma = -0.5*np.log(2*np.pi*self.IPD_var)
        Azimuth_array = mat1['Azimuth_array']
        self.Azimuth_array = Azimuth_array.reshape(Azimuth_array.size, ).tolist()


        self.halfNFFT = self.IPD_mean.shape[1]
        self.halfNFFTtrim = self.halfNFFT-2
        self.Dim_out = self.halfNFFTtrim
        self.Dim_in = self.halfNFFTtrim
        self.channel = 3
        self.batchLen = 100 # each sequence contains 100 frames

        self.miniTrainN = 0
        self.miniValidN = 0

        self.threshold = 25 # energy lower than this (max(energy)-threshold) will be thresholded, for both targets and mixture
        self.EMBEDDINGS_DIM = 40
        threshold2 = 10  # log(x)^2 =10, x = 150.  energy lower than this (max(energy)-threshold) will be treated as silence
        # threshold2 -= self.sig_mean
        threshold2 /= self.sig_std
        self.threshold2 = threshold2

    def TrainDataParamsInit(self, Train_DIR='/SomeDirectoryToChange/DLTrainingData06Mar/'):
        self.Train_DIR = Train_DIR

        TotalFilenameList = list()
        for gender_level in ['MM', 'MF', 'FF']:
            dirName = self.Train_DIR + '{}'.format(gender_level)
            angleList = os.listdir(dirName)
            for ang in angleList:
                fileDirName = dirName + '/{}'.format(ang)
                filenameList = os.listdir(fileDirName)
                TotalFilenameList += filenameList

        TotalSeqNum = len(TotalFilenameList)
        # The training data will be divided to two part, train and valid
        trainNum = int(round(TotalSeqNum * 0.8))
        validNum = TotalSeqNum - trainNum

        print('The total number of training data is ' + str(TotalSeqNum) + ', of which ' +
              str(trainNum) + ' are used for training and ' + str(validNum) + ' for validation.')

        # shuffle the list
        random.seed(999999999)  # Note here, the train and valid should not be muddled up at different training sessions
        shuffle(TotalFilenameList)


        self.trainList = TotalFilenameList[0:trainNum]
        self.validList = TotalFilenameList[trainNum:]

        # # The following num constraints are only for debugging purposes, comment out in real training
        # self.trainList = TotalFilenameList[0:int(self.BATCH_SIZE_Train)]
        # self.validList = TotalFilenameList[self.BATCH_SIZE_Train*16:self.BATCH_SIZE_Train*16+int(self.BATCH_SIZE_Valid*0.5)]

        random.seed(self.seedNum)
        self.validNum = len(self.validList)
        self.trainNum = len(self.trainList)

        print('shuffle the train and valid data')
        shuffle(self.trainList)
        shuffle(self.validList)

        print(self.trainList[0], self.validList[0])

    def processOneBatchFromMat(self, mat, filename, batchFlag=True, mode=1): # This mat data is for training, contains groundtruth IPD shift and dry signals


        mixAngle_LR = np.asarray(mat['mixAngle_LmR'])
        # mixAngle_R = mixAngle_R[:, 0:self.batchLen]

        mixLogPower = np.asarray(mat['mixLogPower'])
        currentThreshold = np.max(mixLogPower) - self.threshold
        mixLogPower[mixLogPower < currentThreshold] = currentThreshold
        # mixLogPower = mixLogPower[:, 0:self.batchLen]

        s1LogPower = np.asarray(mat['s1LogPower'])
        currentThreshold = np.max(s1LogPower) - self.threshold
        s1LogPower[s1LogPower < currentThreshold] = currentThreshold

        s2LogPower = np.asarray(mat['s2LogPower'])
        currentThreshold = np.max(s2LogPower) - self.threshold
        s2LogPower[s2LogPower < currentThreshold] = currentThreshold


        Nums = list(map(int, re.findall('[+-]?\d+', filename)))
        azi_index = self.Azimuth_array.index(Nums[0])  # np.where(self.Azimuth_array == Nums[0])[0][0] angle of the second speaker
        # shiftAng1 = [mixAngle_LR[:,k]-self.IPD_mean[azi1_index] for k in range(mixLogPower.shape[1])]
        shiftAng1 = mixAngle_LR.copy()

        shiftAng2 = mixAngle_LR.copy()
        for k in range(mixLogPower.shape[1]):
            shiftAng2[:, k] -= self.IPD_mean[azi_index]


        shiftAng2[shiftAng2 > np.pi] = shiftAng2[shiftAng2 > np.pi] - 2 * np.pi
        shiftAng2[shiftAng2 <= -np.pi] = shiftAng2[shiftAng2 <= -np.pi] + 2 * np.pi

        shiftAng1 = np.power(shiftAng1, 2)
        shiftAng2 = np.power(shiftAng2, 2)

        shiftAng1 = np.exp(-shiftAng1)
        shiftAng2 = np.exp(-shiftAng2)

        # # normalise the data(prewhitening) with frequency dependent normalisation params
        # for k in range(mixLogPower.shape[1]):
        #     s1LogPower[:, k] -= self.sig_mean
        #     s1LogPower[:, k] /= self.sig_std
        #     mixLogPower[:, k] -= self.sig_mean
        #     mixLogPower[:, k] /= self.sig_std

        # normalise the data(prewhitening) with global normalisation params
        mixLogPower -= self.sig_mean
        mixLogPower /= self.sig_std
        s1LogPower -= self.sig_mean
        s1LogPower /= self.sig_std
        s2LogPower -= self.sig_mean
        s2LogPower /= self.sig_std

        if batchFlag:
            # randomly choose batchLen=100 consecutive frames in each sequence
            # chooseIndex = np.random.randint(0, mixLogPower.shape[1]-self.batchLen, 32)
            try:
                chooseIndex = np.random.randint(0, mixLogPower.shape[1] - self.batchLen + 1)
            except:  # for short sequences that cannot yield 100 frames
                pass
        else:
            N = mixLogPower.shape[1] - self.batchLen + 1
            chooseIndex = np.arange(0, N, self.batchLen)
            if chooseIndex[-1] != N - 1:
                chooseIndex = np.concatenate([chooseIndex, np.asarray([N - 1])], axis=0)

        # concatenate the feature as the input
        Index = range(chooseIndex, chooseIndex+self.batchLen)
        mixLogPower = mixLogPower[:, Index] # (129,T)--->(129,100)
        shiftAng1 = shiftAng1[:, Index] # (129,100)
        shiftAng2 = shiftAng2[:, Index]


        if mode==1:
            # RNN with MSE loss
            currentBatchDataIn = np.concatenate((mixLogPower[1:-1].transpose(), shiftAng1[1:-1].transpose(), shiftAng2[1:-1].transpose()), axis=1) #(100, 127*3)
            currentBatchDataOut = (s1LogPower[1:-1, Index] > s2LogPower[1:-1, Index]).transpose().astype(float) #(100, 127)

        elif mode==2:
            # RNN with perceptually weighted loss
            currentBatchDataIn = np.concatenate( (mixLogPower[1:-1].transpose(), shiftAng1[1:-1].transpose(), shiftAng2[1:-1].transpose()), axis=1)  # (100, 127*3)
            currentBatchDataOut1 = ((s1LogPower[1:-1, Index] > s2LogPower[1:-1, Index]).astype(float)).transpose() #(100, 127)
            currentBatchDataOut2 = np.squeeze(mixLogPower[1:-1]).transpose() # (100,127)
            currentBatchDataOut3 = s1LogPower[1:-1, Index].transpose()  # (100,127)
            currentBatchDataOut = np.concatenate((currentBatchDataOut1, currentBatchDataOut2, currentBatchDataOut3), axis=1) #(100, 127*3)

        elif mode==3:
            # RNN for deep clustering with affinitykmeans loss
            currentBatchDataIn = np.concatenate((mixLogPower[1:-1].transpose(), shiftAng1[1:-1].transpose(), shiftAng2[1:-1].transpose()), axis=1)  # (100, 127*3)

            # Get dominant spectra indexes, create one-hot outputs
            Y = np.zeros((self.batchLen, self.Dim_out) + (2,))

            # # STFTs for individual signals
            # specs = []
            # specs.append(s1LogPower[1:-1, Index].transpose())
            # specs.append(s2LogPower[1:-1, Index].transpose())
            # specs = np.array(specs)
            specs = np.stack((s1LogPower[1:-1, Index].transpose(),s2LogPower[1:-1, Index].transpose()), axis=0)
            vals = np.argmax(specs, axis=0)
            for i in range(2):
                t = np.zeros(2)
                t[i] = 1
                Y[vals == i] = t

            # Create mask for zeroing out gradients from silence components
            X = mixLogPower[1:-1].transpose()
            currentThreshold = np.max(X) - self.threshold2
            z = np.zeros(2)
            Y[X < currentThreshold] = z
            Y = np.reshape(Y,(self.batchLen * self.Dim_out, 2))
            currentBatchDataOut = Y


        return (currentBatchDataIn, currentBatchDataOut, chooseIndex)



    def TrainDataGenerator(self, mode=3):
        # global train_i, trainList
        # Be careful about the data type!
        # # size (number_sample, FrequencyNum, 11, ChannelNum)
        # BatchDataIn = np.zeros((self.BATCH_SIZE_Train, self.batchLen, self.Dim_in*self.channel), dtype='f')
        # # size (number_sample, FrequencyNum)
        # BatchDataOut = np.zeros((self.BATCH_SIZE_Train, self.batchLen, self.Dim_out, 2), dtype='f') # the labels target or interference

        if mode==1:
            # This generates features for RNN in the temporal domain with MSE loss
            # size (number_sample, self.batchLen, FrequencyNum*ChannelNum) 100 x (127*3)
            BatchDataIn = np.zeros((self.BATCH_SIZE_Train, self.batchLen, self.Dim_in*self.channel), dtype='f')
            # size (number_sample, self.batchLen, FrequencyNum)  100 127
            BatchDataOut = np.zeros((self.BATCH_SIZE_Train, self.batchLen, self.Dim_out), dtype='f')

        elif mode==2:
            # RNN, perceptually weighted loss
            # size (number_sample, self.batchLen, FrequencyNum*ChannelNum) 100 x (127*3)
            BatchDataIn = np.zeros((self.BATCH_SIZE_Train, self.batchLen, self.Dim_in*self.channel), dtype='f')
            # size (number_sample, self.batchLen, FrequencyNum)  100 x (127*3)
            BatchDataOut = np.zeros((self.BATCH_SIZE_Train, self.batchLen, self.Dim_out*3), dtype='f')
        elif mode==3:
            # RNN, deep clustering
            # size (number_sample, self.batchLen, FrequencyNum*ChannelNum) 100 x (127*3)
            BatchDataIn = np.zeros((self.BATCH_SIZE_Train, self.batchLen, self.Dim_in*self.channel), dtype='f')
            BatchDataOut = np.zeros((self.BATCH_SIZE_Train, self.batchLen * self.Dim_out, 2), dtype='f') # The label indicating a tf point belongs to target or not

        while 1:
            self.miniTrainN += 1
            if self.verbose:
                print('\nNow collect a mini batch for training----{}'.format(str(self.miniTrainN)))

            i = 0

            while i < self.batchSeqN_Train:

                failN = 0
                while True:
                    try:
                        if self.train_i >= self.trainNum:
                            self.train_i = 0
                            shuffle(self.trainList)
                            if self.verbose:
                                print('Run through all the training data, and shuffle the data again!')
                                print(self.trainList[0])

                        filename = self.trainList[self.train_i]
                        subFolder = filename.split('_')[0]
                        folderName = self.Train_DIR + filename.split('_')[0] + '/' + filename.split('_')[1][3:]
                        mat = sio.loadmat(folderName + '/' + filename)
                        # NotLoadFlag = False
                        # we want to extract the same batchLen frames for each sequence,
                        mixLogPower = np.asarray(mat['mixLogPower'])
                        if mixLogPower.shape[1] >= self.batchLen:
                            break
                        else:
                            self.train_i += 1
                    except:
                        failN += 1
                        time.sleep(failN)  # This is very important, otherwise, the failure persists
                        # if (failN > 1) & (failN % 5 == 0):
                        #     global sio
                        #     import scipy.io as sio

                        if self.verbose:
                            print('Failed {} times, try to load the next sequence ---- {} '.format(failN, self.train_i))
                        self.train_i += 1

                # (100, 127*3) and (100,127,2)
                (currentBatchDataIn, currentBatchDataOut, _) = self.processOneBatchFromMat(mat, filename, True, mode)
                BatchDataIn[i] = currentBatchDataIn
                BatchDataOut[i] = currentBatchDataOut

                self.train_i += 1
                i += 1
            # print(self.valid_i)

            if self.verbose:
                print('Training batch data {} collected'.format(self.miniTrainN))

            yield ([BatchDataIn], [BatchDataOut])

    def ValidDataGenerator(self, mode=3):
        # global train_i, trainList
        # Be careful about the data type!
        # BatchDataIn = np.zeros((self.BATCH_SIZE_Valid, self.batchLen, self.Dim_in*self.channel), dtype='f')
        # BatchDataOut = np.zeros((self.BATCH_SIZE_Valid, self.batchLen, self.Dim_out, 2), dtype='f')  # both the mask and lp of the original signals

        if mode == 1:
            # This generates features for RNN in the temporal domain with MSE loss
            # size (number_sample, self.batchLen, FrequencyNum*ChannelNum) 100 x (127*3)
            BatchDataIn = np.zeros((self.BATCH_SIZE_Valid, self.batchLen, self.Dim_in * self.channel), dtype='f')
            # size (number_sample, self.batchLen, FrequencyNum)  100 127
            BatchDataOut = np.zeros((self.BATCH_SIZE_Valid, self.batchLen, self.Dim_out), dtype='f')

        elif mode==2:
            # RNN, perceptually weighted loss
            # size (number_sample, self.batchLen, FrequencyNum*ChannelNum) 100 x (127*3)
            BatchDataIn = np.zeros((self.BATCH_SIZE_Valid, self.batchLen, self.Dim_in * self.channel), dtype='f')
            # size (number_sample, self.batchLen, FrequencyNum)  100 x (127*3)
            BatchDataOut = np.zeros((self.BATCH_SIZE_Valid, self.batchLen, self.Dim_out*3), dtype='f')

        elif mode==3:
            # RNN, deep clustering
            # size (number_sample, self.batchLen, FrequencyNum*ChannelNum) 100 x (127*3)
            BatchDataIn = np.zeros((self.BATCH_SIZE_Train, self.batchLen, self.Dim_in * self.channel), dtype='f')
            # size (number_sample, self.batchLen, FrequencyNum)  100 x (127*3)
            BatchDataOut = np.zeros((self.BATCH_SIZE_Train, self.batchLen * self.Dim_out, 2), dtype='f')  # The label indicating a tf point belongs to target or not

        while 1:
            self.miniValidN += 1
            if self.verbose:
                print('\nNow collect a mini batch for Validation----{}'.format(str(self.miniValidN)))

            i = 0

            while i < self.batchSeqN_Valid:

                failN = 0
                while True:
                    try:
                        if self.valid_i >= self.validNum:
                            self.valid_i = 0
                            shuffle(self.validList)
                            if self.verbose:
                                print('Run through all the training data, and shuffle the data again!')
                                print(self.validList[0])

                        filename = self.validList[self.valid_i]
                        subFolder = filename.split('_')[0]
                        folderName = self.Train_DIR + filename.split('_')[0] + '/' + filename.split('_')[1][3:]
                        mat = sio.loadmat(folderName + '/' + filename)
                        # NotLoadFlag = False
                        # we want to extract the same batchLen frames for each sequence,
                        mixLogPower = np.asarray(mat['mixLogPower'])
                        if mixLogPower.shape[1] >= self.batchLen:
                            break
                        else:
                            self.valid_i += 1
                    except:
                        failN += 1
                        time.sleep(failN)  # This is very important, otherwise, the failure persists
                        # if (failN > 1) & (failN % 5 == 0):
                        #     global sio
                        #     import scipy.io as sio
                        if self.verbose:
                            print('Failed {} times, try to load the next sequence ---- {} '.format(failN, self.valid_i))
                        self.valid_i += 1

                # (255,100,3) and (255,300)
                (currentBatchDataIn, currentBatchDataOut, _) = self.processOneBatchFromMat(mat, filename, True, mode)
                BatchDataIn[i] = currentBatchDataIn
                BatchDataOut[i] = currentBatchDataOut

                self.valid_i += 1
                i += 1
            # print(self.valid_i)

            if self.verbose:
                print('Validation batch data {} collected'.format(self.miniValidN))
            yield ([BatchDataIn], [BatchDataOut])



# if __name__=="__main__":
#     # One example of data generator :)
#     from keras.datasets import mnist
#     from keras.utils import np_utils
#     # Here shows an example of data generator
#     def myGenerator():
#         # input image dimensions
#         img_rows, img_cols = 28, 28
#         (X_train, y_train), (X_test, y_test) = mnist.load_data()
#         y_train = np_utils.to_categorical(y_train,10)
#         X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#         X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#         X_train = X_train.astype('float32')
#         X_test = X_test.astype('float32')
#         X_train /= 255
#         X_test /= 255
#         while 1:
#             for i in range(1875): # 1875 * 32 = 60000 -> # of training samples
#                 if i%125==0:
#                     print "i = " + str(i)
#                 return X_train[i*32:(i+1)*32], y_train[i*32:(i+1)*32]


if __name__=="__main__":

    dg = dataGenBig()
    dg.TrainDataParamsInit()
    ([aa],[bb]) = dg.ValidDataGenerator(3)# change yield to return to debug the generator

    # visualise the signal
    import matplotlib.pyplot as plt

    index = 4 #
    plt.figure(100)
    plt.title('Training input...')
    ax1 = plt.subplot(311)
    im1 = ax1.pcolor(aa[index,:,0:127].squeeze().transpose())
    plt.colorbar(im1)
    ax2 = plt.subplot(312, sharex=ax1)
    im2 = ax2.pcolor(aa[index,:,127:257].squeeze().transpose())
    plt.colorbar(im2)
    ax3 = plt.subplot(313, sharex=ax1)
    im3 = ax3.pcolor(aa[index, :, 257::].squeeze().transpose())
    plt.colorbar(im3)
    # plt.show()

    # plt.figure(101)
    # plt.title('Training output...')
    # ax1 = plt.subplot(311)
    # im1 = ax1.pcolor(bb[index, :, 0:127].squeeze().transpose())
    # plt.colorbar(im1)
    # ax2 = plt.subplot(312, sharex=ax1)
    # im2 = ax2.pcolor(bb[index, :, 127:257].squeeze().transpose())
    # plt.colorbar(im2)
    # ax3 = plt.subplot(313, sharex=ax1)
    # im3 = ax3.pcolor(bb[index, :, 257::].squeeze().transpose())
    # plt.colorbar(im3)
    # plt.show()

    plt.figure(101)
    plt.title('Training output...')
    ax1 = plt.subplot(211)
    im1 = ax1.pcolor(bb[index, :, :, 0].squeeze().transpose())
    plt.colorbar(im1)
    ax2 = plt.subplot(212, sharex=ax1)
    im2 = ax2.pcolor(bb[index, :, :, 1].squeeze().transpose())
    plt.colorbar(im2)
    plt.show()

    aaa=1
