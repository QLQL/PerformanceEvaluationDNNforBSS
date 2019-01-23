import numpy as np
import scipy.io as sio
import os
from random import shuffle
import random
import time
import re






def tauInit(mat, filename):
    batchLen = 11
    halfBatchLen = 5

    mixAngle_LR = np.asarray(mat['mixAngle_LmR'])
    # mixAngle_R = mixAngle_R[:, 0:self.batchLen]

    mixLogPower = np.asarray(mat['mixLogPower'])
    # mixLogPower = mixLogPower[:, 0:self.batchLen]

    normGCC = np.exp(1j*mixAngle_LR)


    # np.exp(1j*)
    N = 41
    Fs = 16000
    tau_candi = np.linspace(-0.001, 0.001, num=N)
    f_array1 = np.linspace(0,Fs/2, 257)
    f_array = f_array1[1:-1]
    f_array = f_array.reshape((f_array.size, 1))
    c_array = np.zeros((N,))

    # try to use only half of the values
    medianV = np.median(mixLogPower)
    useFlag = np.greater(mixLogPower,medianV)

    for i in range(N):
        tau = tau_candi[i]
        ejwt = np.exp(1j*2*np.pi*f_array*tau)
        # b.reshape((b.size, 1))
        tmp = np.multiply(normGCC,ejwt)
        tmp = np.multiply(tmp,useFlag[1:-1,:])
        c_array[i] = np.mean(tmp)

    # plt.figure(99).suptitle('GCC-PHAT over candidate delays')
    # plt.plot(tau_candi,c_array)
    # plt.grid(True)
    # plt.ylabel('tau')
    # plt.ylabel('GCC')
    # # plt.show()

    # Find the two peaks as the two initial taus associated with the targets
    # first peak
    ind1 = np.argmax(c_array)
    c_array[max(ind1-3,0):min(ind1+3,N)]=0

    # plt.plot(tau_candi, c_array)
    # plt.show()

    # second peak
    ind2 = np.argmax(c_array)

    # signal more to the left side comes out first
    if ind1>ind2:
        tt = ind1
        ind1 = ind2
        ind2 = tt

    peak_tau_array = np.array([tau_candi[ind1],tau_candi[ind2]])
    tmp1 = -2 * np.pi * f_array * tau_candi[ind1]
    tmp1 = tmp1.reshape((1, tmp1.size))
    tmp1 = (tmp1 + np.pi) % (2 * np.pi) - np.pi

    tmp2 = -2 * np.pi * f_array * tau_candi[ind2]
    tmp2 = tmp2.reshape((1, tmp2.size))
    tmp2 = (tmp2 + np.pi) % (2 * np.pi) - np.pi

    IPD_init_array = np.concatenate((tmp1,tmp2),0)

    # plt.figure(99)
    # plt.plot(IPD_init_array[0])
    # plt.plot(IPD_init_array[1])
    # mat1 = sio.loadmat('/vol/vssp/ucdatasets/s3a/DatasetDL_SS/Params/RoomCGMMParams.mat')
    # IPD_mean = mat1['IPD_mean']
    # IPD_mean = np.asarray(IPD_mean).transpose()
    # plt.plot(IPD_mean[0]) #-60
    # plt.plot(IPD_mean[1]) #-30
    # plt.show()

    return (peak_tau_array, IPD_init_array, mixAngle_LR, mixLogPower)


class dataGenBig:
    def __init__(self, seedNum = 123456789, verbose = False):
        self.seedNum = seedNum
        self.verbose = verbose
        # For each sequence, randomly chose 8 feature blocks
        self.NeachSeq = 8

        self.BATCH_SIZE_Train = 128 #32 # 128 #4096  # mini-batch size
        self.batchSeqN_Train = int(round(self.BATCH_SIZE_Train / self.NeachSeq))
        self.BATCH_SIZE_Train = self.batchSeqN_Train * self.NeachSeq

        self.BATCH_SIZE_Valid = 1024  # 32 # 128 #4096  # mini-batch size
        self.batchSeqN_Valid = int(round(self.BATCH_SIZE_Valid / self.NeachSeq))
        self.BATCH_SIZE_Valid = self.batchSeqN_Valid * self.NeachSeq

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
        self.batchLen = 11 # each sequence contains 100 frames
        self.halfBatchLen = int((self.batchLen - 1) / 2)

        self.miniTrainN = 0
        self.miniValidN = 0

        self.threshold = 25 # energy lower than this (max(energy)-threshold) will be thresholded, for both targets and mixture

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

    def processOneBatchFromMat(self, mat, filename, batchFlag=True): # This mat data is for training, contains groundtruth IPD shift and dry signals


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
            # randomly choose N=8 feature vectors in each sequence
            # chooseIndex = np.random.randint(0, mixLogPower.shape[1]-self.batchLen, 32)
            N = self.NeachSeq
            try:
                a = np.arange(mixLogPower.shape[1] - self.batchLen)
                shuffle(a)
                chooseIndex = a[:N]
            except:  # for short sequences that cannot yield 32 none-duplicated data
                chooseIndex = random.randint(0, mixLogPower.shape[1] - self.batchLen, N)
        else:
            N = mixLogPower.shape[1] - self.batchLen + 1
            chooseIndex = np.arange(N)  # (range(0, N)

        # concatenate the feature as the input
        Index1 = (np.tile(range(0, self.batchLen), (N, 1))).transpose()
        Index2 = np.tile(chooseIndex, (self.batchLen, 1))
        Index = Index1 + Index2
        mixLogPower = mixLogPower[:, Index] # (257,T)--->(257,11,8)
        # tmp = np.reshape(tmp, (self.Dim_in, N), order="F")
        mixLogPower = np.transpose(mixLogPower, (2, 0, 1)) #(SampleNumb, 257, 11)

        shiftAng1 = shiftAng1[:, Index] # (257,11,8)
        shiftAng1 = np.transpose(shiftAng1, (2, 0, 1)) # (SampleNumb, 257, 11)

        shiftAng2 = shiftAng2[:, Index]
        shiftAng2 = np.transpose(shiftAng2, (2, 0, 1)) # (SampleNumb, 257, 11)

        currentBatchDataIn = np.stack((mixLogPower[:, 1:-1, :], shiftAng1[:, 1:-1, :], shiftAng2[:, 1:-1, :]), axis=-1) #(SampleNumb, 255, 11, 3)
        #currentBatchDataOut = s1LogPower[1:-1, chooseIndex + self.halfBatchLen].transpose() #(SampleNumb, 255)
        currentBatchDataOut1 = ((s1LogPower[1:-1, chooseIndex + self.halfBatchLen] > s2LogPower[1:-1, chooseIndex + self.halfBatchLen]).astype(float)).transpose()
        currentBatchDataOut2 = np.squeeze(mixLogPower[:, 1:-1, 5])
        currentBatchDataOut3 = s1LogPower[1:-1, chooseIndex + self.halfBatchLen].transpose()  # (SampleNumb, 255)
        currentBatchDataOut = np.concatenate((currentBatchDataOut1, currentBatchDataOut2, currentBatchDataOut3), axis=1)

        return (currentBatchDataIn, currentBatchDataOut, chooseIndex)



    def TrainDataGenerator(self):
        # global train_i, trainList
        # Be careful about the data type!
        # size (number_sample, FrequencyNum, 11, ChannelNum)
        BatchDataIn = np.zeros((self.BATCH_SIZE_Train, self.Dim_in, self.batchLen, self.channel), dtype='f')
        # size (number_sample, FrequencyNum)
        BatchDataOut = np.zeros((self.BATCH_SIZE_Train, self.Dim_out*3), dtype='f') # both the mask and lp of the original signals

        while 1:
            self.miniTrainN += 1
            if self.verbose:
                print('\nNow collect a mini batch for training----{}'.format(str(self.miniTrainN)))

            i = 0
            N = self.NeachSeq

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

                #(SampleNum,255,11,3) and (SampleNum,255)
                (currentBatchDataIn, currentBatchDataOut, _) = self.processOneBatchFromMat(mat, filename)
                BatchDataIn[i * N: (i + 1) * N] = currentBatchDataIn
                BatchDataOut[i * N: (i + 1) * N] = currentBatchDataOut

                self.train_i += 1
                i += 1
            # print(self.valid_i)

            if self.verbose:
                print('Training batch data {} collected'.format(self.miniTrainN))

            yield ([BatchDataIn], [BatchDataOut])

    def ValidDataGenerator(self):
        # global train_i, trainList
        # Be careful about the data type!
        BatchDataIn = np.zeros((self.BATCH_SIZE_Valid, self.Dim_in, self.batchLen, self.channel), dtype='f')
        BatchDataOut = np.zeros((self.BATCH_SIZE_Valid, self.Dim_out*3), dtype='f')

        while 1:
            self.miniValidN += 1
            if self.verbose:
                print('\nNow collect a mini batch for Validation----{}'.format(str(self.miniValidN)))

            i = 0
            N = 8

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

                (currentBatchDataIn, currentBatchDataOut, _) = self.processOneBatchFromMat(mat, filename)
                BatchDataIn[i * N: (i + 1) * N] = currentBatchDataIn
                BatchDataOut[i * N: (i + 1) * N] = currentBatchDataOut

                self.valid_i += 1
                i += 1
            # print(self.valid_i)

            if self.verbose:
                print('Validation batch data {} collected'.format(self.miniValidN))
            yield ([BatchDataIn], [BatchDataOut])



# One example of data generator :)
from keras.datasets import mnist
from keras.utils import np_utils
# Here shows an example of data generator
def myGenerator():
    # input image dimensions
    img_rows, img_cols = 28, 28
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    y_train = np_utils.to_categorical(y_train,10)
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    while 1:
        for i in range(1875): # 1875 * 32 = 60000 -> # of training samples
            if i%125==0:
                print "i = " + str(i)
            yield X_train[i*32:(i+1)*32], y_train[i*32:(i+1)*32]