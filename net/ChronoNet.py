import tensorflow.keras as kr
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model

## Based on source: https://github.com/aguscerdo/EE239AS-Project

def net(config, inception=True, res=True, strided=True, maxpool=False, avgpool=False, batchnorm=True):
    assert (hasattr(config, 'data_format') and
            hasattr(config, 'fs') and
            hasattr(config, 'frame') and
            hasattr(config, 'CH') and
            hasattr(config, 'dropoutRate'))

    input_shape = (config.frame * config.fs, config.CH)

    pad = 'same'
    padp = 'same'
    config.state_size = 32
    config.filters = 32
    config.strides = 2
    config.c_act = 'relu'
    config.r_act = 'sigmoid'
    config.rk_act = 'tanh'

    config.rec_drop = 0
    config.cnn_drop = 0.6

    r = kr.regularizers.l2(config.l2)
    stride_size = config.strides if strided else 1

    input = Input(input_shape)

    if inception:
        c0 = layers.Conv1D(config.filters, kernel_size=2, strides=stride_size, padding=pad,
                           activation=config.c_act)(input)
        c1 = layers.Conv1D(config.filters, kernel_size=4, strides=stride_size, padding=pad,
                           activation=config.c_act)(input)
        c2 = layers.Conv1D(config.filters, kernel_size=8, strides=stride_size, padding=pad,
                           activation=config.c_act)(input)

        c = layers.concatenate([c0, c1, c2])

        if maxpool:
            c = layers.MaxPooling1D(2, padding=padp)(c)
        elif avgpool:
            c = layers.AveragePooling1D(2, padding=padp)(c)
        if batchnorm:
            c = layers.BatchNormalization()(c)

        c = layers.SpatialDropout1D(config.cnn_drop)(c)


        c0 = layers.Conv1D(config.filters, kernel_size=2, strides=stride_size, padding=pad,
                           activation=config.c_act)(c)
        c1 = layers.Conv1D(config.filters, kernel_size=4, strides=stride_size, padding=pad,
                           activation=config.c_act)(c)
        c2 = layers.Conv1D(config.filters, kernel_size=8, strides=stride_size, padding=pad,
                           activation=config.c_act)(c)

        c = layers.concatenate([c0, c1, c2])

        if maxpool:
            c = layers.MaxPooling1D(2, padding=padp)(c)
        elif avgpool:
            c = layers.AveragePooling1D(2, padding=padp)(c)
        if batchnorm:
            c = layers.BatchNormalization()(c)

        c = layers.SpatialDropout1D(config.cnn_drop)(c)


        c0 = layers.Conv1D(config.filters, kernel_size=2, strides=stride_size, padding=pad,
                           activation=config.c_act)(c)
        c1 = layers.Conv1D(config.filters, kernel_size=4, strides=stride_size, padding=pad,
                           activation=config.c_act)(c)
        c2 = layers.Conv1D(config.filters, kernel_size=8, strides=stride_size, padding=pad,
                           activation=config.c_act)(c)

        c = layers.concatenate([c0, c1, c2])

        if maxpool:
            c = layers.MaxPooling1D(2, padding=padp)(c)
        elif avgpool:
            c = layers.AveragePooling1D(2, padding=padp)(c)
        if batchnorm:
            c = layers.BatchNormalization()(c)

        c = layers.SpatialDropout1D(config.cnn_drop)(c)


    else:  # No inception Modules
        c = layers.Conv1D(config.filters, kernel_size=4, strides=stride_size, padding=pad, activation=config.c_act)(
            input)
        if maxpool:
            c = layers.MaxPooling1D(2, padding=padp)(c)
        elif avgpool:
            c = layers.AveragePooling1D(2, padding=padp)(c)
        if batchnorm:
            c = layers.BatchNormalization()(c)
        c = layers.SpatialDropout1D(config.cnn_drop)(c)

        c = layers.Conv1D(config.filters, kernel_size=4, strides=stride_size, padding=pad, activation=config.c_act)(c)
        if maxpool:
            c = layers.MaxPooling1D(2, padding=padp)(c)
        elif avgpool:
            c = layers.AveragePooling1D(2, padding=padp)(c)
        if batchnorm:
            c = layers.BatchNormalization()(c)
        c = layers.SpatialDropout1D(config.cnn_drop)(c)

        c = layers.Conv1D(config.filters, kernel_size=4, strides=stride_size, padding=pad, activation=config.c_act)(c)
        if maxpool:
            c = layers.MaxPooling1D(2, padding=padp)(c)
        elif avgpool:
            c = layers.AveragePooling1D(2, padding=padp)(c)
        if batchnorm:
            c = layers.BatchNormalization()(c)
        c = layers.SpatialDropout1D(config.cnn_drop)(c)

    if res:  # Residual RNN
        g1 = layers.GRU(config.state_size, return_sequences=True, activation=config.rk_act,
                        recurrent_activation=config.r_act, dropout=config.rec_drop, recurrent_dropout=config.rec_drop,
                        recurrent_regularizer=r, kernel_regularizer=r)(c)
        g2 = layers.GRU(config.state_size, return_sequences=True, activation=config.rk_act,
                        recurrent_activation=config.r_act, dropout=config.rec_drop, recurrent_dropout=config.rec_drop,
                        recurrent_regularizer=r, kernel_regularizer=r)(g1)

        g_concat1 = layers.concatenate([g1, g2])

        g3 = layers.GRU(config.state_size, return_sequences=True, activation=config.rk_act, recurrent_activation=config.r_act,
                        dropout=config.rec_drop, recurrent_dropout=config.rec_drop,
                        recurrent_regularizer=r, kernel_regularizer=r)(g_concat1)

        g_concat2 = layers.concatenate([g1, g2, g3])

        g = layers.GRU(config.state_size, return_sequences=False, activation=config.rk_act, recurrent_activation=config.r_act,
                       dropout=config.rec_drop, recurrent_dropout=config.rec_drop,
                       recurrent_regularizer=r, kernel_regularizer=r)(g_concat2)

    else:  # No Residual RNN
        g = layers.GRU(config.state_size, return_sequences=True, activation=config.rk_act, recurrent_activation=config.r_act,
                       dropout=config.rec_drop, recurrent_dropout=config.rec_drop,
                       recurrent_regularizer=r, kernel_regularizer=r)(c)

        g = layers.GRU(config.state_size, return_sequences=True, activation=config.rk_act, recurrent_activation=config.r_act,
                       dropout=config.rec_drop, recurrent_dropout=config.rec_drop,
                       recurrent_regularizer=r, kernel_regularizer=r)(g)
        g = layers.GRU(config.state_size, return_sequences=True, activation=config.rk_act, recurrent_activation=config.r_act,
                       dropout=config.rec_drop, recurrent_dropout=config.rec_drop,
                       recurrent_regularizer=r, kernel_regularizer=r)(g)

        g = layers.GRU(config.state_size, return_sequences=False, activation=config.rk_act, recurrent_activation=config.r_act,
                       dropout=config.rec_drop, recurrent_dropout=config.rec_drop,
                       recurrent_regularizer=r, kernel_regularizer=r)(g)

    d = layers.Dense(2)(g)
    out = layers.Softmax()(d)

    model = Model(input, out)

    return model

