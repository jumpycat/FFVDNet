# verify the influence of basemodels
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_efficientnets import EfficientNetB0, preprocess_input
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Dense, Input, Activation, Conv2D, MaxPooling2D, Flatten, AveragePooling2D, UpSampling2D, \
    concatenate, \
    Subtract, Dot, dot, Multiply, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, Dropout, Lambda, \
    add
from keras.layers.core import Lambda
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.utils import plot_model
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
import os
from tools import *
from keras.models import load_model
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras import layers
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only,
                                                      mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)


def WriteIN(history, path):
    file_write_obj = open(path, 'w')
    _keys = history.history.keys()
    for key in _keys:
        file_write_obj.writelines(key + ':' + str(history.history[key]))
        file_write_obj.write('\n')
    file_write_obj.close()


def Xception(img_rows, img_cols, color_type, weights_path='../models/xception_weights_tf_dim_ordering_tf_kernels.h5'):
    img_input = Input(shape=(img_rows, img_cols, color_type))

    # Block 1
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 2
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Block 2 Pool
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 3
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Block 3 Pool
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 4
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # Block 5 - 12
    for i in range(8):
        residual = x

        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)

        x = layers.add([x, residual])

    residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 13
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Block 13 Pool
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # Block 14
    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Block 14 part 2
    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Fully Connected Layer
    x_fc = GlobalAveragePooling2D()(x)
    x_fc = Dense(1000, activation='softmax')(x_fc)

    inputs = img_input

    # Create model
    model = Model(inputs, x_fc, name='xception')

    # load weights
    model.load_weights(weights_path)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = GlobalAveragePooling2D()(x)
    x_newfc = LeakyReLU(0.1)(x_newfc)
    x_newfc = Dense(1, activation='sigmoid', name='fc_out')(x_newfc)

    # Create another model with our customized softmax
    model = Model(img_input, x_newfc)

    return model


def _EffnetB0():
    model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    input = Input((224, 224, 3))
    x = model(input)
    x = GlobalAveragePooling2D()(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=x)
    model.summary()
    plot_model(model, to_file='effnet.png', show_shapes=True)
    return model


def _InceptionV3():
    model = InceptionV3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    input = Input((224, 224, 3))
    x = model(input)
    x = GlobalAveragePooling2D()(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=x)
    model.summary()
    plot_model(model, to_file='inceptionv3.png', show_shapes=True)
    return model


def _mobilenetv1():
    model = MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    input = Input((224, 224, 3))
    x = model(input)
    x = GlobalAveragePooling2D()(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=x)
    model.summary()
    # plot_model(model,to_file='inceptionv3.png',show_shapes=True)
    return model


def _mobilenetv2():
    model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    input = Input((224, 224, 3))
    x = model(input)
    x = GlobalAveragePooling2D()(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=x)
    model.summary()
    # plot_model(model,to_file='inceptionv3.png',show_shapes=True)
    return model


def train(model, epoch, _name):
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer=SGD(0.01, momentum=0.9, decay=0.0095 / epoch))
    train_datagen = ImageDataGenerator(
        rescale=1 / 255.0,
        # horizontal_flip = True
    )
    #
    test_datagen = ImageDataGenerator(
        rescale=1 / 255.0
    )

    train_gen = train_datagen.flow_from_directory(r'/Data/olddata_E/01DeepFakesDetection/04FF++_Full_Videos/BigImagesWithMask/DeepFake/train', target_size=(224, 224),
                                                  batch_size=32, class_mode='binary')
    val_gen = test_datagen.flow_from_directory(r'/Data/olddata_E/01DeepFakesDetection/04FF++_Full_Videos/BigImagesWithMask/DeepFake/val', target_size=(224, 224),
                                               batch_size=32, class_mode='binary')

    filepath = r"../models/DeepFake/" + _name + r"-weights-improvement-{epoch:03d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False,
                                 mode='auto')
    print(train_gen.class_indices)
    history = model.fit_generator(train_gen, steps_per_epoch=50, validation_steps=100,
                                  validation_data=val_gen, epochs=epoch, verbose=1, callbacks=[checkpoint])
    WriteIN(history, r'../records/DeepFake' + _name + r'_224_0-40.txt')

if __name__ == "__main__":
    model = _EffnetB0()
    # model = _mobilenetv1()
    # model = Xception(224,224,3)
    #model = _InceptionV3()
    train(model, 40, 'efficientnetb0')
