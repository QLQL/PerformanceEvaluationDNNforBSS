from keras import layers, models, optimizers, initializers
from keras import backend as K
import tensorflow as tf

import numpy as np
from keras.regularizers import l2



def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.

    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    """

    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
        inputs_expand = K.expand_dims(inputs, 1)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule]
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        # x.shape=[num_capsule, input_num_capsule, input_dim_capsule]
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # Regard the first two dimensions as `batch` dimension,
        # then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, self.input_num_capsule].
        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape=[batch_size, num_capsule, input_num_capsule]
            c = tf.nn.softmax(b, dim=1)

            # c.shape =  [batch_size, num_capsule, input_num_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # The first two dimensions as `batch` dimension,
            # then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
            # outputs.shape=[None, num_capsule, dim_capsule]
            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))  # [None, 10, 16]

            if i < self.routings - 1:
                # outputs.shape =  [None, num_capsule, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
                # b.shape=[batch_size, num_capsule, input_num_capsule]
                b += K.batch_dot(outputs, inputs_hat, [2, 3])
        # End: Routing algorithm -----------------------------------------------------------------------#

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])


class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional
    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
    masked Tensor.
    For example:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
        out = Mask()(x)  # out.shape=[8, 6]
        # or
        out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
        ```
    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # generate the mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])




def GenerateBSSCNNNet():
    channels = 3
    spectrum_height = 127
    spectrum_width = 11
    input_shape = (spectrum_height, spectrum_width, channels)

    spectrum_tensor = layers.Input(shape=input_shape)

    ########### encoder
    # Layer 1: A conventional Conv2D layer with batch noarmalisation and max pooling
    conv1 = layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='valid', name='conv1')(spectrum_tensor)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.LeakyReLU()(conv1)
    conv1 = layers.pooling.MaxPooling2D(pool_size=(2, 2), padding='valid')(conv1)

    # Layer 2: Another conventional Conv2D layer with batch noralisation and maxpooling
    conv2 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='conv2')(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.LeakyReLU()(conv2)
    conv2 = layers.pooling.MaxPooling2D(pool_size=(2, 1), padding='valid')(conv2)

    # Layer 3: Another conventional Conv2D layer with batch noralisation and maxpooling
    conv3 = layers.Conv2D(filters=256, kernel_size=(8, 1), strides=(1, 1), padding='valid', name='conv3')(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.LeakyReLU()(conv3)
    encoded = layers.pooling.MaxPooling2D(pool_size=(4, 1), padding='valid')(conv3)

    pre_output = layers.Flatten()(encoded)

    # ########### decoder

    # Decoder model
    decoder = models.Sequential(name='PostProcessDense')
    decoder.add(layers.Dense(1024, input_dim=pre_output._keras_shape[1]))
    decoder.add(layers.BatchNormalization())
    decoder.add(layers.LeakyReLU())
    decoder.add(layers.Dense(spectrum_height, activation='sigmoid'))

    spectrum_output = decoder(pre_output)

    BSS_model = models.Model(inputs=spectrum_tensor, outputs=spectrum_output)

    return BSS_model


def GenerateBSSFullNet():

    # spectrum_tensor, input_shape, pre_output, _ = preProcess()
    channels = 3
    spectrum_height = 127
    spectrum_width = 11
    input_shape = (spectrum_height, spectrum_width, channels)

    spectrum_tensor = layers.Input(shape=input_shape)

    ##########################################################################################
    # a fully connected layer

    pre_output = layers.Flatten()(spectrum_tensor)

    # Decoder model
    denseNet = models.Sequential(name='DenseLayer')
    denseNet.add(layers.Dense(1024, input_dim=pre_output._keras_shape[1]))
    denseNet.add(layers.BatchNormalization())
    denseNet.add(layers.LeakyReLU())

    denseNet.add(layers.Dense(1024))
    denseNet.add(layers.BatchNormalization())
    denseNet.add(layers.LeakyReLU())

    denseNet.add(layers.Dense(1024))
    denseNet.add(layers.BatchNormalization())
    denseNet.add(layers.LeakyReLU())

    denseNet.add(layers.Dense(1024))
    denseNet.add(layers.BatchNormalization())
    denseNet.add(layers.LeakyReLU())

    denseNet.add(layers.Dense(spectrum_height, activation='sigmoid'))

    spectrum_output = denseNet(pre_output)

    ##########################################################################################

    BSS_model = models.Model(inputs=spectrum_tensor, outputs=spectrum_output)

    return BSS_model


def GenerateBSSCapsNet():

    def preProcess():
        channels = 3
        spectrum_height = 127
        spectrum_width = 11
        input_shape = (spectrum_height, spectrum_width, channels)

        spectrum_tensor = layers.Input(shape=input_shape)

        conv1 = models.Sequential(name='PreProcessConv')

        # Layer 1: A conventional Conv2D layer with batch noarmalisation and max pooling
        conv1.add(layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='valid', input_shape=input_shape))
        conv1.add(layers.BatchNormalization())
        conv1.add(layers.LeakyReLU())
        conv1.add(layers.pooling.MaxPooling2D(pool_size=(2, 2), padding='valid'))

        conv1.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        conv1.add(layers.BatchNormalization())
        conv1.add(layers.LeakyReLU())
        conv1.add(layers.pooling.MaxPooling2D(pool_size=(2, 1), padding='valid'))

        # conv1.add(layers.Conv2D(filters=256, kernel_size=(4, 4), strides=(1, 1), padding='valid'))
        # conv1.add(layers.BatchNormalization())
        # conv1.add(layers.LeakyReLU())
        # conv1.add(layers.pooling.MaxPooling2D(pool_size=(2, 1), padding='valid'))

        conv1_output = conv1(spectrum_tensor)

        return spectrum_tensor, input_shape, conv1_output, conv1

    def postProcess(input, spectrum_height):
        # Decoder model
        decoder = models.Sequential(name='PostProcessDense')
        decoder.add(layers.Dense(1024, input_dim=input._keras_shape[1]))
        decoder.add(layers.BatchNormalization())
        decoder.add(layers.LeakyReLU())
        # decoder.add(layers.Dense(512))
        # decoder.add(layers.BatchNormalization())
        # decoder.add(layers.LeakyReLU())
        # output mask
        decoder.add(layers.Dense(spectrum_height, activation='sigmoid'))

        spectrum_output = decoder(input)

        return [spectrum_output, decoder]

    spectrum_tensor, input_shape, pre_output, preProcessModel = preProcess()

    ##########################################################################################
    # capsule model
    capsmodel = models.Sequential(name='capsmodel')

    capsmodel.add(layers.Reshape(target_shape=[-1, 8],input_shape=pre_output._keras_shape[1:]))
    capsmodel.add(layers.Lambda(squash, name='primarycap'))
    capsmodel.add(CapsuleLayer(num_capsule=16, dim_capsule=64, routings=3, name='digitcaps'))

    digitcaps = capsmodel(pre_output)
    out_caps = Length(name='capsnetlength')(digitcaps)
    y = Mask(name='capsnetmask')(digitcaps)


    ##########################################################################################
    [spectrum_output, decoder] = postProcess(y, input_shape[0])

    BSS_model = models.Model(inputs=[spectrum_tensor], outputs=[spectrum_output])

    return BSS_model



if __name__=="__main__":
    train_model = GenerateBSSCapsNet()
    print(train_model.summary())
    from keras.utils import plot_model
    plot_model(train_model, to_file='CapsNetModel.png')






# def preProcessOld():
#     channels = 3
#     spectrum_height = 255
#     spectrum_width = 11
#     input_shape = (spectrum_height, spectrum_width, channels)
#
#     spectrum_tensor = layers.Input(shape=input_shape)
#
#     # Layer 1: A conventional Conv2D layer with batch noarmalisation and max pooling
#     conv1 = layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 1), padding='valid', name='conv1')(spectrum_tensor)
#     conv1 = layers.BatchNormalization()(conv1)
#     conv1 = layers.LeakyReLU()(conv1)
#     conv1 = layers.pooling.MaxPooling2D(pool_size=(2, 1), padding='valid')(conv1)
#
#     # Layer 2: Another conventional Conv2D layer with batch noralisation and maxpooling
#     conv2 = layers.Conv2D(filters=128, kernel_size=(4, 4), strides=(1, 1), padding='valid', name='conv2')(conv1)
#     conv2 = layers.BatchNormalization()(conv2)
#     conv2 = layers.LeakyReLU()(conv2)
#     conv2 = layers.pooling.MaxPooling2D(pool_size=(2, 1), padding='valid')(conv2)
#
#     # Layer 3: Another conventional Conv2D layer with batch noralisation and maxpooling
#     conv3 = layers.Conv2D(filters=256, kernel_size=(4, 4), strides=(1, 1), padding='valid', name='conv3')(conv2)
#     conv3 = layers.BatchNormalization()(conv3)
#     conv3 = layers.LeakyReLU()(conv3)
#     conv3 = layers.pooling.MaxPooling2D(pool_size=(2, 1), padding='valid')(conv3)
#
#     # # Layer 4: Another conventional Conv2D layer with batch noralisation and maxpooling
#     # conv4 = layers.Conv2D(filters=128, kernel_size=(2, 2), strides=(1, 1), padding='valid', name='conv4')(conv3)
#     # conv4 = layers.BatchNormalization()(conv4)
#     # conv4 = layers.LeakyReLU()(conv4)
#     # conv4 = layers.pooling.MaxPooling2D(pool_size=(2, 1), padding='valid')(conv4)
#
#     return spectrum_tensor, spectrum_height, conv3








