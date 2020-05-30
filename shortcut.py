import cv2
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_efficientnets import EfficientNetB0, preprocess_input
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Dense, Input, Activation, Conv2D, MaxPooling2D, Flatten, AveragePooling2D, UpSampling2D, \
    concatenate, \
    Subtract, Dot, dot, Multiply, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, Dropout, Lambda, \
    add, GlobalAvgPool2D, Conv3D, DepthwiseConv2D
from keras.layers.core import Lambda
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.layers.advanced_activations import LeakyReLU
# from MakeData import myGenerator, MakeValidation, GenRatio, MakeValidationDiff, MakeValidationSame,GenSeq,_GenVal
from keras import backend as K
from keras.utils import plot_model
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
import os
import random
import time
# from tools import *
from keras.layers import Dense, Flatten, Dropout, TimeDistributed, Input, Conv2D, LSTM, GlobalAvgPool2D, LeakyReLU
from config import *
import math
import tensorflow as tf
from keras.engine.topology import Layer
from keras.models import load_model
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, add

FRAME_NUM = 12
HEIGHT = 224
WIDTH = 224
CHANNEL = 3

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Swish(Layer):

    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, training=None):
        return tf.nn.swish(inputs)


def get_default_block_list():
    DEFAULT_BLOCK_LIST = [
        BlockArgs(32, 16, kernel_size=3, strides=(1, 1), num_repeat=1, se_ratio=0.25, expand_ratio=1),
        BlockArgs(16, 24, kernel_size=3, strides=(2, 2), num_repeat=2, se_ratio=0.25, expand_ratio=6),
        BlockArgs(24, 40, kernel_size=5, strides=(2, 2), num_repeat=2, se_ratio=0.25, expand_ratio=6),
        BlockArgs(40, 80, kernel_size=3, strides=(2, 2), num_repeat=3, se_ratio=0.25, expand_ratio=6),
        BlockArgs(80, 112, kernel_size=5, strides=(1, 1), num_repeat=3, se_ratio=0.25, expand_ratio=6),
        BlockArgs(112, 192, kernel_size=5, strides=(2, 2), num_repeat=4, se_ratio=0.25, expand_ratio=6),
        BlockArgs(192, 320, kernel_size=3, strides=(1, 1), num_repeat=1, se_ratio=0.25, expand_ratio=6),
    ]
    return DEFAULT_BLOCK_LIST


block_args_list = get_default_block_list()


def round_filters(filters, width_coefficient, depth_divisor, min_depth):
    """Round number of filters based on depth multiplier."""
    multiplier = float(width_coefficient)
    divisor = int(depth_divisor)
    min_depth = min_depth

    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor

    return int(new_filters)


class DropConnect(Layer):

    def __init__(self, drop_connect_rate=0., **kwargs):
        super(DropConnect, self).__init__(**kwargs)
        self.drop_connect_rate = float(drop_connect_rate)

    def call(self, inputs, training=None):
        def drop_connect():
            keep_prob = 1.0 - self.drop_connect_rate

            # Compute drop_connect tensor
            batch_size = tf.shape(inputs)[0]
            random_tensor = keep_prob
            random_tensor += K.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
            binary_tensor = tf.floor(random_tensor)
            output = (inputs / keep_prob) * binary_tensor
            return output

        return K.in_train_phase(drop_connect, inputs, training=training)

    def get_config(self):
        config = {
            'drop_connect_rate': self.drop_connect_rate,
        }
        base_config = super(DropConnect, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def round_repeats(repeats, depth_coefficient):
    """Round number of filters based on depth multiplier."""
    multiplier = depth_coefficient

    if not multiplier:
        return repeats

    return int(math.ceil(multiplier * repeats))


def SEBlock(input_filters, se_ratio, expand_ratio, data_format=None):
    if data_format is None:
        data_format = K.image_data_format()

    num_reduced_filters = max(
        1, int(input_filters * se_ratio))
    filters = input_filters * expand_ratio

    if data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    def block(inputs):
        x = inputs
        x = Lambda(lambda a: K.mean(a, axis=spatial_dims, keepdims=True))(x)
        x = Conv2D(
            num_reduced_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding='same',
            use_bias=True)(x)
        x = Swish()(x)
        # Excite
        x = Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding='same',
            use_bias=True)(x)
        x = Activation('sigmoid')(x)
        out = Multiply()([x, inputs])
        return out

    return block


def MBConvBlock(input_filters, output_filters,
                kernel_size, strides,
                expand_ratio, se_ratio,
                id_skip, drop_connect_rate,
                batch_norm_momentum=0.99,
                batch_norm_epsilon=1e-3,
                data_format=None):
    if data_format is None:
        data_format = K.image_data_format()

    if data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    has_se = (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1)
    filters = input_filters * expand_ratio

    def block(inputs):

        if expand_ratio != 1:
            x = Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding='same',
                use_bias=False)(inputs)
            x = BatchNormalization(
                axis=channel_axis,
                momentum=batch_norm_momentum,
                epsilon=batch_norm_epsilon)(x)
            x = Swish()(x)
        else:
            x = inputs

        x = DepthwiseConv2D(
            [kernel_size, kernel_size],
            strides=strides,
            padding='same',
            use_bias=False)(x)
        x = BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)(x)
        x = Swish()(x)

        if has_se:
            x = SEBlock(input_filters, se_ratio, expand_ratio,
                        data_format)(x)

        # output phase

        x = Conv2D(
            output_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding='same',
            use_bias=False)(x)
        x = BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)(x)

        if id_skip:
            # if all(s == 1 for s in strides) and (
            #         input_filters == output_filters):
            #
            #     # only apply drop_connect if skip presents.
            #     if drop_connect_rate:
            #         x = DropConnect(drop_connect_rate)(x)
            #     x = Add()([x, inputs])
            pass
        return x

    return block


def Efficientnetb0(input_shape=(224, 224, 3),
                   width_coefficient=1.0,
                   depth_coefficient=1.0,
                   drop_connect_rate=0.,
                   batch_norm_momentum=0.99,
                   batch_norm_epsilon=1e-3,
                   depth_divisor=8,
                   min_depth=None):
    data_format = K.image_data_format()
    channel_axis = -1
    block_args_list = get_default_block_list()

    # count number of strides to compute min size
    stride_count = 1
    for block_args in block_args_list:
        if block_args.strides is not None and block_args.strides[0] > 1:
            stride_count += 1

    min_size = int(2 ** stride_count)

    # Stem part
    inputs = Input(shape=input_shape)

    x = inputs
    x = Conv2D(
        filters=round_filters(32, width_coefficient,
                              depth_divisor, min_depth),
        kernel_size=[3, 3],
        strides=[2, 2],
        padding='same',
        use_bias=False)(x)
    x = BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon)(x)
    x = Swish()(x)

    num_blocks = sum([block_args.num_repeat for block_args in block_args_list])
    drop_connect_rate_per_block = drop_connect_rate / float(num_blocks)

    # Blocks part
    for block_idx, block_args in enumerate(block_args_list):
        assert block_args.num_repeat > 0

        # Update block input and output filters based on depth multiplier.
        block_args.input_filters = round_filters(block_args.input_filters, width_coefficient, depth_divisor, min_depth)
        block_args.output_filters = round_filters(block_args.output_filters, width_coefficient, depth_divisor,
                                                  min_depth)
        block_args.num_repeat = round_repeats(block_args.num_repeat, depth_coefficient)

        # The first block needs to take care of stride and filter size increase.
        x = MBConvBlock(block_args.input_filters, block_args.output_filters,
                        block_args.kernel_size, block_args.strides,
                        block_args.expand_ratio, block_args.se_ratio,
                        block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                        batch_norm_momentum, batch_norm_epsilon, data_format)(x)

        if block_args.num_repeat > 1:
            block_args.input_filters = block_args.output_filters
            block_args.strides = [1, 1]

        for _ in range(block_args.num_repeat - 1):
            x = MBConvBlock(block_args.input_filters, block_args.output_filters,
                            block_args.kernel_size, block_args.strides,
                            block_args.expand_ratio, block_args.se_ratio,
                            block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                            batch_norm_momentum, batch_norm_epsilon, data_format)(x)

    # Head part
    x = Conv2D(
        filters=round_filters(1280, width_coefficient, depth_coefficient, min_depth),
        kernel_size=[1, 1],
        strides=[1, 1],
        padding='same',
        use_bias=False)(x)
    x = BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon)(x)
    x = Swish()(x)

    # if include_top:
    #     x = GlobalAveragePooling2D(data_format=data_format)(x)
    #
    #     if dropout_rate > 0:
    #         x = Dropout(dropout_rate)(x)
    #
    #     x = Dense(classes)(x)
    #     x = Activation('softmax')(x)
    #
    # else:
    #     if pooling == 'avg':
    #         x = GlobalAveragePooling2D()(x)
    #     elif pooling == 'max':
    #         x = GlobalMaxPooling2D()(x)

    x = GlobalAveragePooling2D()(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(1, activation='sigmoid')(x)

    outputs = x
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.

    model = Model(inputs, outputs)
    return model


# fake and real totally         video_n = 0 represents all
def ReadPath_shuffle(path, video_n):
    path_list = []
    for root, dirs, files in os.walk(path):
        for file in dirs:
            path_list.append(os.path.join(root, file))
    path_list.remove(path_list[0])
    path_list.remove(path_list[0])
    random.shuffle(path_list)
    if video_n == 0:
        return path_list
    else:
        return path_list[:video_n]  # return paths of video_frame folder


def ReadPic_shuffle(pth):
    file_list = []
    for root, dirs, files in os.walk(pth):
        # files.sort(key= lambda x:int(x[:-4]))
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list  # paths of frames in video_frame folder


def CreatData_shuffle(path, video_n, frame_n, mode, rate):  # model represents video or image classification
    print("Loading Data")
    time_ = time.time()
    if mode == 0:
        video_frame = []  # 2-D list[video[frame]]
        label = []
        path_list = ReadPath_shuffle(path, video_n)
        for pth in path_list:
            pths = ReadPic_shuffle(pth)
            video_frame.append(pths)
        video_n = len(video_frame)
        img = cv2.imread(video_frame[0][0])
        shape_1, shape_2, shape_3 = img.shape
        data = np.zeros((video_n * rate, frame_n, shape_1, shape_2, shape_3))

        for i in range(video_n):
            for m in range(rate):
                if 'original' in video_frame[i][0]:
                    label.append(1)
                else:
                    label.append(0)
                x = random.randint(0, len(video_frame[i]) - frame_n)
                for j in range(frame_n):
                    img = cv2.imread(video_frame[i][x + j]) / 255.0
                    data[i * rate + m][j][:, :, :] = img[:, :, :]
        # label = np.array(label)
        print("Data Loaded")
        print(time.time() - time_)
        return data, np.array(label)

    if mode == 1:
        video_frame_total = []  # 1-D list[video_frames]
        label = []
        path_list = ReadPath_shuffle(path, video_n)

        for pth in path_list:
            pths = ReadPic_shuffle(pth)
            num_f = len(pths)
            x = random.sample(range(0, num_f), frame_n)
            picked_frames = type(path_list)(map(lambda m: pths[m], x))
            video_frame_total += picked_frames

        random.shuffle(video_frame_total)
        image_n = len(video_frame_total)
        data = np.zeros((image_n, 224, 224, 3))
        print("image number: " + str(image_n))

        for i in range(image_n):
            frame_path = video_frame_total[i]
            img = cv2.imread(frame_path) / 255.0
            img = cv2.resize(img, (224, 224))
            data[i][:, :, :] = img[:, :, :]
            if 'original' in frame_path:
                label.append(1)
            else:
                label.append(0)
        label = np.array(label)
        print(label)
        print("Data Loaded")
        print(time.time() - time_)
        return data, label


def myGeneratorFrame(pth, batch_size):
    print("Generator Initializing")
    path_list = ReadPath_shuffle(pth, 0)
    video_num = len(path_list)

    frame_list = []
    for index in range(video_num):
        video_frames = ReadPic_shuffle(path_list[index])
        frame_list += video_frames
    random.shuffle(frame_list)
    length = len(frame_list)
    start_index = 0
    print("Generator Loaded")
    while True:
        # data = np.zeros((batch_size, frame_length, 224, 224, 3))
        data = []
        label = []

        end_index = start_index + batch_size
        picked_frames = frame_list[start_index:end_index]
        start_index = end_index

        if start_index + batch_size > length:
            random.shuffle(frame_list)
            start_index = 0

        for (i, frame) in enumerate(picked_frames):
            if "original" in frame:
                label.append(1)
            else:
                label.append(0)
            data.append(cv2.imread(picked_frames[i]) / 255.0)
        yield np.array(data), np.array(label)


class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only,
                                                      mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)


def _EffnetB0():
    model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    input = Input((224, 224, 3))
    x = model(input)
    x = GlobalAveragePooling2D()(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=x)
    model.summary()
    # plot_model(model,to_file='effnet.png',show_shapes=True)
    return model


# if __name__ == "__main__":
#     batch_size = 32
#     base = Efficientnetb0()
#     base.load_weights(r'../models/efficientnet-b0_notop.h5', by_name=True)
#     base.summary()
#     # plot_model(base, 'model.png', show_shapes=True)

#     filepath = r"../models/nt_base/weights-improvement-{epoch:03d}-{val_acc:.4f}.hdf5"
#     checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=False, mode='auto')

#     base.compile(optimizer=SGD(lr=0.01,momentum=0.9,decay=0.0004), loss='binary_crossentropy', metrics=['accuracy'])
#     GEN = myGeneratorFrame(r"E:\01DeepFakesDetection\04FF++_Full_Videos\Images\DF_crop224\train",batch_size)
#     val_data,val_label= CreatData_shuffle(r'E:\01DeepFakesDetection\04FF++_Full_Videos\Images\DF_crop224\val',280,5,1,0)#path,video_n,frame_n,mode,rate


#     base.fit_generator(generator=GEN,steps_per_epoch=50,epochs=20,validation_data=(val_data,val_label),verbose=1,callbacks=[checkpoint])

def WriteIN(history, path):
    file_write_obj = open(path, 'w')
    _keys = history.history.keys()
    for key in _keys:
        file_write_obj.writelines(key + ':' + str(history.history[key]))
        file_write_obj.write('\n')
    file_write_obj.close()


def Xception(img_rows, img_cols, color_type):
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
    # x = add([x, residual])

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
    # x = add([x, residual])

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
    # x = add([x, residual])

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

    # x = add([x, residual])

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
    # x = add([x, residual])

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
    # model.load_weights(weights_path)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = GlobalAveragePooling2D()(x)
    x_newfc = LeakyReLU(0.1)(x_newfc)
    x_newfc = Dense(1, activation='sigmoid', name='fc_out')(x_newfc)

    # Create another model with our customized softmax
    model = Model(img_input, x_newfc)

    return model


def train(model, epoch):
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

    train_gen = train_datagen.flow_from_directory(
        r'/Data/olddata_E/01DeepFakesDetection/04FF++_Full_Videos/BigImagesWithMask/DeepFake/train',
        target_size=(224, 224), batch_size=32, class_mode='binary')
    val_gen = test_datagen.flow_from_directory(
        r'/Data/olddata_E/01DeepFakesDetection/04FF++_Full_Videos/BigImagesWithMask/DeepFake/val',
        target_size=(224, 224), batch_size=32, class_mode='binary')

    filepath = r"../models/deepfake/xception-without-shortcut-weights-improvement-{epoch:03d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False,
                                 mode='auto')
    print(train_gen.class_indices)
    history = model.fit_generator(train_gen, class_weight={0: 1., 1: 1.}, steps_per_epoch=50, validation_steps=100,
                                  validation_data=val_gen, epochs=epoch, verbose=1, callbacks=[checkpoint])
    WriteIN(history, r'../records/df_xception-wtihout-shortcut_224_0-40.txt')


if __name__ == "__main__":
    # model = Efficientnetb0()
    model = Xception(224, 224, 3)

    train(model, 40)
