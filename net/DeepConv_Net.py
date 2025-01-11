
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l2


def net(config):
    """ Adapted Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.

    This implementation assumes the input is a 2-second EEG signal sampled at
    128Hz, as opposed to signals sampled at 250Hz as described in the original
    paper. We also perform temporal convolutions of length (1, 5) as opposed
    to (1, 10) due to this sampling rate difference.

    Note that we use the max_norm constraint on all convolutional layers, as
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication
    with the original authors.

                      ours        original paper
    pool_size        1, 2        1, 3
    strides          1, 2        1, 3
    conv filters     1, 5        1, 10

    Note that this implementation has not been verified by the original
    authors.

    """
    kernel_const = 5
    assert (hasattr(config, 'data_format') and
            hasattr(config, 'fs') and
            hasattr(config, 'frame') and
            hasattr(config, 'CH') and
            hasattr(config, 'dropoutRate'))

    if hasattr(config, 'l2'):
        reg_set = {'kernel_regularizer': l2(config.l2),
                    'bias_regularizer': l2(config.l2)}

    input_shape = (config.CH, config.frame*config.fs, 1)

    i = Input(shape=input_shape)

    # start the model
    block1 = Conv2D(25, (1, 5), input_shape=input_shape, kernel_constraint=max_norm(kernel_const, axis=(0, 1, 2)),
                    kernel_initializer="he_normal", **reg_set)(i)
    block1 = Conv2D(25, (config.CH, 1), kernel_constraint=max_norm(kernel_const, axis=(0, 1, 2)),
                    kernel_initializer="he_normal", **reg_set)(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1 = Dropout(config.dropoutRate)(block1)

    block2 = Conv2D(50, (1, 5), kernel_constraint=max_norm(kernel_const, axis=(0, 1, 2)),
                    kernel_initializer="he_normal", **reg_set)(block1)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2 = Dropout(config.dropoutRate)(block2)

    block3 = Conv2D(100, (1, 5), kernel_constraint=max_norm(kernel_const, axis=(0, 1, 2)),
                    kernel_initializer="he_normal", **reg_set)(block2)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3 = Dropout(config.dropoutRate)(block3)

    block4 = Conv2D(200, (1, 5), kernel_constraint=max_norm(kernel_const, axis=(0, 1, 2)),
                    kernel_initializer="he_normal", **reg_set)(block3)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4 = Dropout(config.dropoutRate)(block4)

    flatten = Flatten()(block4)

    # dense = Dense(config.nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    dense = Dense(2)(flatten)

    out = Activation('softmax')(dense)

    return Model(inputs=i, outputs=out)