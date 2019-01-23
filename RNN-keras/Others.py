import numpy as np
import matplotlib.pyplot as plt
from STFT import istft
import scipy.io.wavfile
import scipy.io as sio
import re

class RecoverData:
    def __init__(self, normParamsDir = '/SomeDirectoryToChange/Params/myNormParamsMixture.mat', sp = 0.5, Fs = 8000):
        self.normParamsDir = normParamsDir
        self.sp = sp
        self.Fs = Fs

        if normParamsDir is not None:
            mat2 = sio.loadmat(normParamsDir)
            # sig_std = mat2['sig_std_w'][1:-1]  # Dimx1
            # sig_mean = mat2['sig_mean_w'][1:-1]  # Dimx1
            # self.sig_std = sig_std.reshape(sig_std.size, )
            # self.sig_mean = sig_mean.reshape(sig_std.size, )
            self.sig_std = np.asscalar(mat2['sig_std'])  # global std
            self.sig_mean = np.asscalar(mat2['sig_mean'])  # global std



    def recover_from_spectrum(self, output, output_angle, savename = 'test.wav'): # size NFFT/2-1 * samples
        if self.normParamsDir is not None:
            # The original spectrum reverse the pre-whitening
            # for k in range(output.shape[0]):
            #     output[k] *= self.sig_std
            #     output[k] += self.sig_mean
            output *= self.sig_std
            output += self.sig_mean

        # now apply istft
        # from log-spectrum to the magnitude
        # output = np.sqrt(np.power(10, output))  # get the magnitude
        output = np.sqrt(np.exp(output))  # get the magnitude
        output = np.multiply(output, np.exp(1j * output_angle))
        output2 = output[:, ::-1].conj()
        tempzero = np.zeros((output.shape[0],1))
        output = np.concatenate((tempzero, output, tempzero, output2), axis=1)
        # plt.figure(101)
        # plt.pcolor(np.abs(output[0:300,:].transpose()))
        # plt.show()

        NFFT = output.shape[1]
        shift = int(self.sp * NFFT)
        x = istft(output, NFFT, shift)
        # plt.figure(102)
        # plt.title('Enhanced signal Wave...')
        # plt.plot(x)

        if savename is not None:
            scaled = np.int16(x / np.max(np.abs(x)) * 32767)
            scipy.io.wavfile.write(savename, self.Fs, scaled)

        return x

def extractInputFeature(mat, filename, mode=1):  # This mat data is for test

    mat2 = sio.loadmat('/SomeDirectoryToChange/Params/myNormParamsMixture.mat')
    # sig_std = mat2['sig_std_w']  # Dimx1
    # sig_mean = mat2['sig_mean_w']  # Dimx1
    # sig_std = sig_std.reshape(sig_std.size, )
    # sig_mean = sig_mean.reshape(sig_std.size, )
    sig_std = np.asscalar(mat2['sig_std'])  # global std
    sig_mean = np.asscalar(mat2['sig_mean'])  # global std

    batchLen = 100

    # calculat the IPD mean of the interfering speaker
    mixAngle_LR = np.asarray(mat['mixAngle_LmR'])
    # mixAngle_R = mixAngle_R[:, 0:self.batchLen]

    mixLogPower = np.asarray(mat['mixLogPower'])
    # mixLogPower = mixLogPower[:, 0:self.batchLen]

    normGCC = np.exp(1j * mixAngle_LR[1:-1])

    # np.exp(1j*)
    N = 41
    Fs = 8000
    tau_candi = np.linspace(-0.001, 0.001, num=N)
    f_array1 = np.linspace(0, Fs / 2, 129)
    f_array = f_array1[1:-1]
    f_array = f_array.reshape((f_array.size, 1))
    c_array = np.zeros((N,))

    # try to use only half of the values
    medianV = np.median(mixLogPower)
    useFlag = np.greater(mixLogPower, medianV)

    for i in range(N):
        tau = tau_candi[i]
        ejwt = np.exp(1j * 2 * np.pi * f_array * tau)
        # b.reshape((b.size, 1))
        tmp = np.multiply(normGCC, ejwt)
        tmp = np.multiply(tmp, useFlag[1:-1])
        c_array[i] = np.mean(tmp)

    # import matplotlib.pyplot  as plt
    # plt.figure(99).suptitle('GCC-PHAT over candidate delays')
    # plt.plot(tau_candi,c_array)
    # plt.grid(True)
    # plt.ylabel('tau')
    # plt.ylabel('GCC')
    # # plt.show()

    # Find the two peaks as the two initial taus associated with the targets
    # first peak
    ind1 = np.argmax(c_array)
    c_array[max(ind1 - 3, 0):min(ind1 + 3, N)] = 0

    # plt.plot(tau_candi, c_array)
    # plt.show()

    # second peak
    ind2 = np.argmax(c_array)

    # signal more to the left side comes out first
    if ind1 > ind2:
        tt = ind1
        ind1 = ind2
        ind2 = tt

    peak_tau_array = np.array([tau_candi[ind1], tau_candi[ind2]])

    peakind = np.argmax(np.abs(peak_tau_array))
    peak_tau = peak_tau_array[peakind]

    tmp1 = -2 * np.pi * f_array1 * peak_tau
    tmp1 = tmp1.reshape((1, tmp1.size))
    IPD_init_array = (tmp1 + np.pi) % (2 * np.pi) - np.pi

    shiftAng1 = mixAngle_LR.copy()
    shiftAng2 = mixAngle_LR.copy()
    for k in range(mixLogPower.shape[1]):
        shiftAng2[:, k] -= IPD_init_array[0]

    shiftAng2[shiftAng2 > np.pi] = shiftAng2[shiftAng2 > np.pi] - 2 * np.pi
    shiftAng2[shiftAng2 <= -np.pi] = shiftAng2[shiftAng2 <= -np.pi] + 2 * np.pi

    shiftAng1 = np.power(shiftAng1, 2)
    shiftAng2 = np.power(shiftAng2, 2)

    shiftAng1 = np.exp(-shiftAng1)
    shiftAng2 = np.exp(-shiftAng2)

    # normalise the data(prewhitening)
    for k in range(mixLogPower.shape[1]):
        mixLogPower[:, k] -= sig_mean
        mixLogPower[:, k] /= sig_std

    N = mixLogPower.shape[1] - batchLen + 1
    chooseIndex = np.arange(0, N, batchLen)
    if chooseIndex[-1] != N-1:
        chooseIndex = np.concatenate([chooseIndex,np.asarray([N-1])],axis=0)

    # concatenate the feature as the input
    Index1 = (np.tile(range(0, batchLen), (len(chooseIndex), 1))).transpose()
    Index2 = np.tile(chooseIndex, (batchLen, 1))
    Index = Index1 + Index2
    mixLogPower = mixLogPower[1:-1, Index]  # (257,T)--->(127,100,SampleNumb)
    # tmp = np.reshape(tmp, (self.Dim_in, N), order="F")
    mixLogPower = np.transpose(mixLogPower, (2, 1, 0))  # (SampleNumb, 100, 127)

    shiftAng1 = shiftAng1[1:-1, Index]  # (127,100,SampleNumb)
    shiftAng1 = np.transpose(shiftAng1, (2, 1, 0))  # (SampleNumb, 100, 127)

    shiftAng2 = shiftAng2[1:-1, Index]
    shiftAng2 = np.transpose(shiftAng2, (2, 1, 0))  # (SampleNumb, 100, 127)

    currentBatchDataIn = np.concatenate((mixLogPower, shiftAng1, shiftAng2), axis=2)  # (SampleNumb, 100, 127*3)

    return (currentBatchDataIn, chooseIndex)




from keras import backend as K
mat2 = sio.loadmat('/SomeDirectoryToChange/Params/myNormParamsMixture.mat')
sig_std = np.asscalar(mat2['sig_std'])  # global std
def my_loss(y_true, y_pred):
    """
    y_true: NumSample x 3W, ( the 3 dimensions are ideal mask, LP_mixture, LP_groundtruth)
    y_pred: NumSamples x W
    """
    # print(y_true.shape)
    # print(y_pred.shape)

    IBM = y_true[:,:,0:127]
    mixtureLP = y_true[:,:,127:254] # normalised
    groundthLP = y_true[:,:,254:381] # normalised

    # estimateLP = (log(mx)^2 - u)/sigma = 2log(m)/sigma + mixtureLP
    estimateLP = (2/sig_std)*K.log(IBM)+mixtureLP

    diff = K.square(y_pred - IBM)

    weight1 = K.sigmoid(groundthLP)
    weight2 = K.sigmoid(estimateLP)

    weight = (1-weight1)*weight2 + weight1

    return K.mean(K.sum(diff*weight, axis=-1), axis = -1)




def affinitykmeans(Y, V):
    # Y size [BATCH_SIZE, 100 x 127, 2] for two signal situations
    # V size [BATCH_SIZE, 100, 127 x EMBEDDINGS_DIMENSION]
    def norm(tensor):
        square_tensor = K.square(tensor)
        frobenius_norm2 = K.sum(square_tensor, axis=(1, 2))
        return frobenius_norm2

    def dot(x, y):
        return K.batch_dot(x, y, axes=(2, 1))

    def T(x):
        return K.permute_dimensions(x, [0, 2, 1])

    # BATCH_SIZE = Y._keras_shape[0]
    MAX_MIX = 2 # either target or inteference
    EMBEDDINGS_DIMENSION = 40 # change this later
    Nframe = 100
    temp = K.reshape(V, [-1, 127*Nframe, EMBEDDINGS_DIMENSION])
    V = K.l2_normalize(temp, axis=-1)

    # Y = K.reshape(Y, [-1, 127*Nframe, MAX_MIX])

    silence_mask = K.sum(Y, axis=2, keepdims=True)
    V = silence_mask * V

    return norm(dot(T(V), V)) - norm(dot(T(V), Y)) * 2 + norm(dot(T(Y), Y))




