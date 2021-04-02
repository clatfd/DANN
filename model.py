import numpy as np
import keras
from keras.layers import Input, Dense, Conv3D, MaxPooling3D, BatchNormalization, Activation, Flatten,Dropout
from keras.models import Model


def build_models(config):
    """Creates three different models, one used for source only training, two used for domain adaptation"""
    inputs = Input(shape=(config['patch_height'], config['patch_width'], config['depth'], config['channel']),
                   name='patchimg')

    kernelinitfun = keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None)
    activationfun = 'relu'
    # kernelinitfun = 'glorot_normal'

    x = Conv3D(32, (3, 3, 3), padding='same', kernel_initializer=kernelinitfun, name='conv1_1')(inputs)
    x = Conv3D(32, (3, 3, 3), padding='same', kernel_initializer=kernelinitfun, name='conv1_2')(x)
    # x = Dropout(0.3)(x)
    x = BatchNormalization(name='bn1')(x)
    x = Activation(activationfun)(x)
    x = MaxPooling3D(name='mp1', strides=(2, 2, 1))(x)
    x = Conv3D(32, (3, 3, 3), padding='same', kernel_initializer=kernelinitfun, name='conv2_1')(x)
    x = Conv3D(32, (3, 3, 3), padding='same', kernel_initializer=kernelinitfun, name='conv2_2')(x)
    # x = Dropout(0.2)(x)
    x = BatchNormalization(name='bn2')(x)
    x = Activation(activationfun)(x)
    x = MaxPooling3D(name='mp2', strides=(2, 2, 1))(x)
    x = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='same', kernel_initializer=kernelinitfun, name='conv3_1')(x)
    x = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='same', kernel_initializer=kernelinitfun, name='conv3_2')(x)
    # x = Dropout(0.5)(x)
    x = BatchNormalization(name='bn3')(x)
    x = Activation(activationfun)(x)
    x = MaxPooling3D(name='mp3', strides=(2, 2, 1))(x)
    x = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='same', kernel_initializer=kernelinitfun, name='conv4_1')(x)
    x = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding='same', kernel_initializer=kernelinitfun, name='conv4_2')(x)
    # x = Dropout(0.5)(x)
    x = BatchNormalization(name='bn4')(x)
    x = Activation(activationfun)(x)

    x4 = Flatten(name='aux_fx')(x)

    source_classifier = Dropout(0.5)(x4)
    source_classifier = Dense(512, activation='softmax', name="mo1")(source_classifier)
    source_classifier = Dropout(0.5)(source_classifier)
    source_classifier = Dense(128, activation='softmax', name="mo2")(source_classifier)
    # source_classifier = Dropout(0.3)(source_classifier)
    source_classifier = Dense(1, name="mo")(source_classifier)

    domain_classifier = Dense(32, activation='linear', name="do4")(x4)
    domain_classifier = BatchNormalization(name="do5")(domain_classifier)
    domain_classifier = Activation("elu", name="do6")(domain_classifier)
    domain_classifier = Dropout(0.5)(domain_classifier)

    domain_classifier = Dense(2, activation='softmax', name="do")(domain_classifier)

    adamop = keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999, amsgrad=False)
    comb_model = Model(inputs=inputs, outputs=[source_classifier, domain_classifier])
    comb_model.compile(optimizer=adamop,
                       loss={'mo': 'mae', 'do': 'categorical_crossentropy'},
                       loss_weights={'mo': 1, 'do': 2}, metrics=['accuracy'], )

    source_classification_model = Model(inputs=inputs, outputs=[source_classifier])
    source_classification_model.compile(optimizer=adamop,
                                        loss={'mo': 'mae'}, metrics=['accuracy'], )

    domain_classification_model = Model(inputs=inputs, outputs=[domain_classifier])
    domain_classification_model.compile(optimizer=adamop,
                                        loss={'do': 'categorical_crossentropy'}, metrics=['accuracy'])

    embeddings_model = Model(inputs=inputs, outputs=[x4])
    embeddings_model.compile(optimizer=adamop, loss='categorical_crossentropy', metrics=['accuracy'])

    return comb_model, source_classification_model, domain_classification_model, embeddings_model


def batch_generator(data, batch_size, config, target=False):
    """Generate batches of data.

    Given a list of numpy data, it iterates over the list and returns batches of the same size
    This
    """
    listsize = min(len(data.pilist), 10)
    while 1:
        nrd = np.arange(len(data.pilist))
        np.random.shuffle(nrd)
        for ri in range(len(data.pilist) // listsize):
            cpilist = [data.pilist[nrd[rid]] for rid in range(listsize * ri, listsize * (ri + 1))]
            data.loadpatch(config, cpilist)
            if target:
                data_l = [data.xarray, np.zeros(shape=(len(data.xarray), 1))]
            else:
                data_l = [data.xarray, data.yarray - 1]

            for di in range(len(data_l[0])):
                data_l[0][di][:, :, :, 0] = data_l[0][di][:, :, :, 0] / np.max(data_l[0][di][:, :, :, 0])

            for repi in range(3):
                data_l = shuffle_aligned_list(data_l)

                batch_count = 0
                while True:
                    if batch_count * batch_size + batch_size >= len(data_l[0]):
                        if len(data.pilist) // listsize != 1:
                            print('list end', ri * listsize)
                            break
                        else:
                            batch_count = 0

                    start = batch_count * batch_size
                    end = start + batch_size
                    batch_count += 1
                    yield [d[start:end] for d in data_l]


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


