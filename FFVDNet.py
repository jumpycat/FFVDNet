"新算法，将7*7*1280的特征图重拍，之后利用3D CNN"

import cv2
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_efficientnets import EfficientNetB0, preprocess_input
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Dense, Input, Activation, Conv2D, MaxPooling2D, Flatten, AveragePooling2D, UpSampling2D, \
    concatenate, \
    Subtract, Dot, dot, Multiply, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, Dropout, Lambda, \
    add, GlobalAvgPool2D, Conv3D, GlobalAveragePooling3D
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
from keras.utils import to_categorical
from keras.utils.multi_gpu_utils import multi_gpu_model
from DepthwiseConv3D import DepthwiseConv3D
from keras.models import load_model


# fake and real totally
def ReadPath_shuffle(path, video_n):
    path_list = []
    for root, dirs, files in os.walk(path):
        for file in dirs:
            path_list.append(os.path.join(root, file))
    path_list.remove(path_list[0])
    path_list.remove(path_list[0])
    random.shuffle(path_list)
    return path_list[:video_n]  # return paths of video_frame folder


def ReadPic_shuffle(pth):
    file_list = []
    for root, dirs, files in os.walk(pth):
        files.sort(key=lambda x: int(x[:-4]))
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list  # paths of frames in video_frame folder


def CreatData_shuffle(path, video_n, frame_n, mode):
    # model represents video or image classification
    # video_n 表示总共读取多少视频
    # frame_n 选取前多少张图像
    if mode == 0:
        video_frame = []  # 2-D list[video[frame]]
        label = []
        path_list = ReadPath_shuffle(path, video_n)
        for pth in path_list:
            if 'original' in pth:
                label.append(1)
            else:
                label.append(0)
            pths = ReadPic_shuffle(pth)
            video_frame.append(pths)
        video_n = len(video_frame)
        data = np.zeros((video_n, frame_n, 224, 224, 3))
        for i in range(video_n):
            for j in range(frame_n):
                img = cv2.imread(video_frame[i][j]) / 255.0
                data[i][j][:, :, :] = img[:, :, :]
        label = np.array(label)
        return data, label

    if mode == 1:
        video_frame_total = []  # 1-D list[video_frames]
        label = []
        path_list = ReadPath_shuffle(path, video_n)

        for pth in path_list:
            pths = ReadPic_shuffle(pth)[:frame_n]
            video_frame_total += pths

        random.shuffle(video_frame_total)
        for frame_pth in video_frame_total:
            if 'original' in frame_pth:
                label.append(1)
            else:
                label.append(0)

        image_n = len(video_frame_total)
        img = cv2.imread(video_frame_total[0])
        shape_1, shape_2, shape_3 = img.shape
        data = np.zeros((image_n, shape_1, shape_2, shape_3))
        print(image_n)
        for i in range(image_n):
            img = cv2.imread(video_frame_total[i]) / 255.0
            data[i][:, :, :] = img[:, :, :]
        label = np.array(label)
        return data, label, image_n, shape_1, shape_2, shape_3


def feature_reset(x):
    x = K.permute_dimensions(x, (0, 4, 2, 3, 1))
    return x


def antirectifier_output_shape(input_shape):
    shape = list(input_shape)
    return tuple(shape)


def make_videoclips_4ff(model='train', batch_size=8, FRAME_NUM=6):
    # 0 = fake, 1=real
    if model == 'train':
        path = r"/Data/olddata_E/01DeepFakesDetection/04FF++_Full_Videos/BigImagesWithMask/DeepFake/train"
    if model == 'test':
        path = r'/Data/olddata_E/01DeepFakesDetection/04FF++_Full_Videos/BigImagesWithMask/DeepFake/test'

    #     path = r"/Data/data1/04FF/DeepFake/DF_Videos/train"
    # if model == 'test':
    #     path = r'/Data/data1/04FF/DeepFake/DF_Videos/val'

    A, B = os.listdir(path)
    path_ = [path + '/' + A, path + '/' + B]

    video_id = [[path_[0] + '/' + i for i in os.listdir(path_[0])], [path_[1] + '/' + j for j in os.listdir(path_[1])]]

    while True:
        data = []
        label = []
        for i in range(batch_size):
            sample = []
            _class = random.randint(0, 1)
            _video = random.randint(0, len(video_id[_class]) - 1)
            # _video = random.randint(0, len(video_id[_class][_person])-1)
            frame_list = (os.listdir(video_id[_class][_video]))
            if '@eaDir' in frame_list:
                frame_list.remove('@eaDir')

            if 'Thumbs.db' in frame_list:
                # os.remove(video_id[_class][_video]+'/'+'Thumbs.db')
                frame_list.remove('Thumbs.db')

            frame_list.sort(key=lambda x: int(x.split('_')[-1][:-4]))
            if len(frame_list) <= FRAME_NUM + 1:
                pass
            else:

                # 连续的
                _start = random.randint(0, len(frame_list) - 1 - FRAME_NUM)
                label.append(_class)
                for k in range(_start, _start + FRAME_NUM):
                    img = cv2.resize(cv2.imread(video_id[_class][_video] + '/' + frame_list[k]), (224, 224)) / 255.0
                    sample.append(img)
                data.append(sample)

            # 不连续的
            # label.append(_class)
            # for k in range(FRAME_NUM):
            #     i = random.randint(0, len(frame_list)-1-FRAME_NUM)
            #     img = cv2.resize(cv2.imread(video_id[_class][_video]+'/'+frame_list[i]),(224,224))/255.0
            #     sample.append(img)
            # data.append(sample)

        yield np.array(data), np.array(label)


def make_videoclips_4dfdc(model='train', batch_size=8, _random=False, FRAME_NUM=6, ):
    # 0 = fake, 1=real
    if model == 'train':
        path = r"/Data/data2/01DFDC_PREVIEW/datafaces/train"
    if model == 'test':
        path = r'/Data/data2/01DFDC_PREVIEW/datafaces/test'

    A, B = os.listdir(path)
    path_ = [path + '/' + A, path + '/' + B]

    video_id = [[path_[0] + '/' + i for i in os.listdir(path_[0])], [path_[1] + '/' + j for j in os.listdir(path_[1])]]

    # video_id = [
    #     [[person_id[0][k]+'/'+m for m in os.listdir(person_id[0][k])] for k  in range(len(person_id[0]))],
    #     [[person_id[1][k]+'/'+m for m in os.listdir(person_id[1][k])] for k  in range(len(person_id[1]))],
    # ]
    # [[fake[person[videoclips[frame[]]]]],[real]]

    # print(len(video_id[1]))

    while True:
        data = []
        label = []
        for i in range(batch_size):
            sample = []
            _class = random.randint(0, 1)
            _video = random.randint(0, len(video_id[_class]) - 1)
            # _video = random.randint(0, len(video_id[_class][_person])-1)
            frame_list = (os.listdir(video_id[_class][_video]))
            if '@eaDir' in frame_list:
                frame_list.remove('@eaDir')

            if 'Thumbs.db' in frame_list:
                os.remove(video_id[_class][_video] + '/' + 'Thumbs.db')
                frame_list.remove('Thumbs.db')
            if not _random:
                frame_list.sort(key=lambda x: int(x.split('_')[-1][:-4]))
            else:
                random.shuffle(frame_list)
            if len(frame_list) <= FRAME_NUM + 1:
                pass
            else:
                _start = random.randint(0, len(frame_list) - 1 - FRAME_NUM)
                label.append(_class)
                for k in range(_start, _start + FRAME_NUM):
                    img = cv2.resize(cv2.imread(video_id[_class][_video] + '/' + frame_list[k]), (224, 224)) / 255.0
                    sample.append(img)
                data.append(sample)
        yield np.array(data), np.array(label)
    # return data,label


def make_videoclips_4celeb_df(model='train', batch_size=8, FRAME_NUM=6):
    # 0 = fake, 1=real
    if model == 'train':
        path = r"/Data/data1/03Celeb-DF_Full_Videos/SPILT_IMG/train"
    if model == 'test':
        path = r'/Data/data1/03Celeb-DF_Full_Videos/SPILT_IMG/test'

    A, B = os.listdir(path)
    path_ = [path + '/' + A, path + '/' + B]

    person_id = [[path_[0] + '/' + i for i in os.listdir(path_[0])], [path_[1] + '/' + j for j in os.listdir(path_[1])]]

    video_id = [
        [[person_id[0][k] + '/' + m for m in os.listdir(person_id[0][k])] for k in range(len(person_id[0]))],
        [[person_id[1][k] + '/' + m for m in os.listdir(person_id[1][k])] for k in range(len(person_id[1]))],
    ]
    # [[fake[person[videoclips[frame[]]]]],[real]]

    # print(len(video_id[1]))

    while True:
        data = []
        label = []
        for i in range(batch_size):
            sample = []
            _class = random.randint(0, 1)
            _person = random.randint(0, len(video_id[_class]) - 1)
            _video = random.randint(0, len(video_id[_class][_person]) - 1)
            frame_list = (os.listdir(video_id[_class][_person][_video]))
            if '@eaDir' in frame_list:
                frame_list.remove('@eaDir')
            if 'Thumbs.db' in frame_list:
                os.remove(video_id[_class][_person][_video] + '/' + 'Thumbs.db')
                frame_list.remove('Thumbs.db')
            frame_list.sort(key=lambda x: int(x.split('_')[-1][:-4]))
            if len(frame_list) <= FRAME_NUM + 1:
                pass
            else:
                _start = random.randint(0, len(frame_list) - 1 - FRAME_NUM)
                label.append(_class)
                for k in range(_start, _start + FRAME_NUM):
                    img = cv2.resize(cv2.imread(video_id[_class][_person][_video] + '/' + frame_list[k]),
                                     (224, 224)) / 255.0
                    sample.append(img)
                data.append(sample)
        yield np.array(data), np.array(label)
    # return data,label


def WriteIN(history, path):
    file_write_obj = open(path, 'w')
    _keys = history.history.keys()
    for key in _keys:
        file_write_obj.writelines(key + ':' + str(history.history[key]))
        file_write_obj.write('\n')
    file_write_obj.close()


class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only,
                                                      mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)


def train(epoch, FRAME_NUM, _num, batch_size, _random=False):
    FRAME_NUM = FRAME_NUM
    HEIGHT = 224
    WIDTH = 224
    CHANNEL = 3

    base = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_ = Model(base.inputs, base.outputs)

    # base = load_model('../models/nt/efficientb0-weights-improvement-027-0.9912.hdf5')
    # base_ = Model(base.get_layer('model_1').inputs, base.get_layer('model_1').outputs)

    seq_input = Input((FRAME_NUM, HEIGHT, WIDTH, CHANNEL))
    x = TimeDistributed(base_)(seq_input)

    # x = DepthwiseConv3D(kernel_size=(3,3,3),strides = (1,1,1),depth_multiplier=1)(x)
    # x = LeakyReLU(0.1)(x)
    # x = DepthwiseConv3D(kernel_size=(4,3,3),strides = (1,1,1),depth_multiplier=1)(x)
    # x = LeakyReLU(0.1)(x)
    # x = DepthwiseConv3D(kernel_size=(3,3,3), depth_multiplier=1)(x)
    # x = LeakyReLU(0.1)(x)
    # x = GlobalAveragePooling3D()(x)
    # x = LeakyReLU(0.1)(x)

    # base method
    x = Lambda(feature_reset)(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3)))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = LeakyReLU(0.1)(x)
    x = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3)))(x)
    x = LeakyReLU(0.1)(x)
    x = TimeDistributed(Conv2D(filters=1, kernel_size=(3, 3)))(x)
    x = LeakyReLU(0.1)(x)
    ##

    # x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)

    model_ = Model(inputs=seq_input, outputs=x, name='FJWnet')

    multi_model = multi_gpu_model(model_, gpus=4)

    model_.summary()
    multi_model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                        optimizer=SGD(0.01, momentum=0.9, decay=0.0095 / epoch))

    train_gen = make_videoclips_4dfdc(model='train', batch_size=batch_size, FRAME_NUM=FRAME_NUM, _random=_random)
    test_gen = make_videoclips_4dfdc(model='test', batch_size=batch_size, FRAME_NUM=FRAME_NUM, _random=_random)

    if _random == False:
        filepath = r"../models/NewIdea/dfdc_f=" + _num + r"/efficientb0-weights-improvement-{epoch:03d}-{val_acc:.4f}.hdf5"
    else:
        filepath = r"../models/NewIdea/dfdc_f=" + _num + r"/random_efficientb0-weights-improvement-{epoch:03d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False,
                                 mode='auto')

    Checkpoint = ParallelModelCheckpoint(model_, filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                         save_weights_only=False, mode='auto')

    history = multi_model.fit_generator(train_gen, class_weight={0: 1., 1: 1.}, steps_per_epoch=100,
                                        validation_steps=200, validation_data=test_gen, epochs=epoch, verbose=1,
                                        callbacks=[Checkpoint])

    if _random == False:
        WriteIN(history, r'../records/dfdc_efficientb0-new_' + _num + r'_frames_224_0-40.txt')
    else:
        WriteIN(history, r'../records/dfdc_efficientb0-shuffled_new_' + _num + r'_frames_224_0-40.txt')


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

# train(40,3,'3')
# train(40,6,'6')
# train(40,12,'12')
# train(60,15,'15',12)
# train(epoch=80,FRAME_NUM=18,_num = '18',batch_size= 8,_random = False)
# train(epoch=60,FRAME_NUM=15,_num = '15',batch_size= 12,_random = True)
train(epoch=40, FRAME_NUM=6, _num='6', batch_size=16, _random=False)
