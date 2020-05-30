import os
import cv2
import numpy as np
import random
from keras.utils import to_categorical
import time
from PIL import Image

#fake and real totally         video_n = 0 represents all
def ReadPath_shuffle(path,video_n):
    path_list = []
    for root,dirs,files in os.walk(path):
        for file in dirs:
            path_list.append(os.path.join(root,file))
    path_list.remove(path_list[0])
    path_list.remove(path_list[0])
    random.shuffle(path_list)
    if video_n==0:
        return path_list
    else:
        return path_list[:video_n] #return paths of video_frame folder

def ReadPic_shuffle(pth):
    file_list = []
    for root,dirs,files in os.walk(pth):
        # files.sort(key= lambda x:int(x[:-4]))
        for file in files:
            file_list.append(os.path.join(root,file))
    return file_list #paths of frames in video_frame folder

def CreatData_shuffle(path,video_n,frame_n,mode,rate):#model represents video or image classification
    print("Loading Data")
    time_ = time.time()
    if mode == 0:
        video_frame = [] #2-D list[video[frame]]
        label = []
        path_list = ReadPath_shuffle(path,video_n)
        for pth in path_list:
            pths = ReadPic_shuffle(pth)
            video_frame.append(pths)
        video_n = len(video_frame)
        img = cv2.imread(video_frame[0][0])
        shape_1,shape_2,shape_3 = img.shape
        data = np.zeros((video_n*rate,frame_n,shape_1,shape_2,shape_3))


        for i in range(video_n):
            for m in range(rate):
                if 'original' in video_frame[i][0]:
                    label.append(1)
                else:
                    label.append(0)
                x = random.randint(0, len(video_frame[i])-frame_n)
                for j in range(frame_n):
                    img = cv2.imread(video_frame[i][x+j])/255.0
                    data[i*rate+m][j][:,:,:] = img[:,:,:]
        # label = np.array(label)
        print("Data Loaded")
        print(time.time()-time_)
        print(data.shape)
        print(len(label))
        return data,np.array(label),frame_n,shape_1,shape_2,shape_3


    if mode == 1:
        video_frame_total = []  # 1-D list[video_frames]
        label = []
        path_list = ReadPath_shuffle(path, video_n)

        for pth in path_list:
            pths = ReadPic_shuffle(pth)
            num_f = len(pths)
            x = random.sample(range(0,num_f),frame_n)
            picked_frames = type(path_list)(map(lambda m:pths[m], x))
            video_frame_total += picked_frames

        random.shuffle(video_frame_total)

        image_n = len(video_frame_total)
        img = cv2.imread(video_frame_total[0])

        # shape_1, shape_2, shape_3 = img.shape
        data = np.zeros((image_n, 224, 224, 3))

        print("image number: "+str(image_n))

        for i in range(image_n):
            frame_path = video_frame_total[i]
            img = cv2.imread(frame_path) / 255.0
            img = cv2.resize(img,(224,224))
            data[i][:, :, :] = img[:, :, :]
            if 'original' in frame_path:
                label.append(1)
            else:
                label.append(0)
        label = np.array(label)
        print(label)
        print("Data Loaded")
        print(time.time()-time_)
        return data, label, image_n

def myGenerator(pth,batch_size,frame_length):
    path_list = ReadPath_shuffle(pth,0)
    video_num = len(path_list)
    # video_frame = ReadPic_shuffle(path_list[0])
    # print(video_frame)

    # a = len(video_frame)
    # print(a)
    start_index = 0
    while True:
        data = np.zeros((batch_size, frame_length, 224, 224, 3))
        label = []

        end_index = start_index+batch_size
        picked_videos = path_list[start_index:end_index]
        start_index = end_index

        if start_index+batch_size>len(path_list):
             random.shuffle(path_list)
             start_index = 0


        # x = random.sample(range(0,video_num),batch_size)
        # picked_videos = type(path_list)(map(lambda i:path_list[i], x))

        for (i,video) in enumerate(picked_videos):
            video_frames = ReadPic_shuffle(video)
            video_length = len(video_frames)
            start = random.randint(0, video_length-frame_length)
            end = start+frame_length
            picked_frames = video_frames[start:end]
            # print(video)
            if "original" in video:
                label.append(1)
            else:
                label.append(0)
            for j in range(frame_length):
                img = cv2.imread(picked_frames[j])/255.0
                data[i][j][:,:,:] = img[:,:,:]
        # print(label)
        yield data,np.array(label)

def ReturnPersonPth():
    path_list = []
    for root, dirs, files in os.walk(r'E:\06Gufei\CroppedYale'):
        for file in dirs:
            path_list.append(os.path.join(root, file))
    random.shuffle(path_list)
    return path_list


# def ReadPic_shuffle(pth):
#     file_list = []
#     for root, dirs, files in os.walk(pth):
#         # files.sort(key= lambda x:int(x[:-4]))
#         for file in files:
#             file_list.append(os.path.join(root, file))
#     return file_list  # paths of frames in video_frame folder


def ReturnData_Label():
    final = []
    for path in ReturnPersonPth():
        final += ReadPic_shuffle(path)
    return final


# def myGenerator(batch_size):
#     height = 192
#     weigth = 168
#     data_all = []
#     person_list = ReturnPersonPth()
#     for path in person_list:
#         file_list = []
#         for root, dirs, files in os.walk(path):
#             for file in files:
#                 file_list.append(os.path.join(root, file))
#         data_all.append(file_list)
#
#     while True:
#         pos_ratio = random.uniform(0.1, 0.9)
#         pos_num = int(batch_size * pos_ratio)
#         # height_ratio = random.uniform(0.8, 1.2)
#         # weigth_ratio = random.uniform(0.9, 1.5)
#         data_1 = []
#         data_2 = []
#         label_1 = []
#         label_2 = []
#         label_3 = []
#         for i in range(pos_num):
#             person_same = random.randint(0, 37)
#             ill_same = random.randint(0, 63)
#             person_same_another = random.randint(0, 37)
#             left = data_all[person_same][ill_same]
#             right = data_all[person_same_another][ill_same]
#
#             img_1 = Image.open(left)
#             img_2 = Image.open(right)
#             img_1 = np.array(img_1)
#             img_2 = np.array(img_2)
#             img_1 = cv2.resize(img_1, (int(weigth * weigth_ratio), int(height * height_ratio)))
#             img_2 = cv2.resize(img_2, (int(weigth * weigth_ratio), int(height * height_ratio)))
#             img_1 = img_1[:, :, np.newaxis] / 255.0
#             img_2 = img_2[:, :, np.newaxis] / 255.0
#             data_1.append(img_1)
#             data_2.append(img_2)
#             label_1.append(0)
#         sampled_list = [n for n in range(64)]
#         for i in range(pos_num, batch_size):
#             person_diff = random.randint(0, 37)
#             ill_diff = random.sample(sampled_list, 2)
#             person_diff_another = random.randint(0, 37)
#             left = data_all[person_diff][ill_diff[0]]
#             right = data_all[person_diff_another][ill_diff[1]]
#             img_1 = Image.open(left)
#             img_2 = Image.open(right)
#             img_1 = np.array(img_1)
#             img_2 = np.array(img_2)
#             img_1 = cv2.resize(img_1, (int(weigth * weigth_ratio), int(height * height_ratio)))
#             img_2 = cv2.resize(img_2, (int(weigth * weigth_ratio), int(height * height_ratio)))
#             img_1 = img_1[:, :, np.newaxis] / 255.0
#             img_2 = img_2[:, :, np.newaxis] / 255.0
#             data_1.append(img_1)
#             data_2.append(img_2)
#             label_1.append(1)
#         yield [np.array(data_1), np.array(data_2)], [np.array(label), np.array(label), np.array(label)]


def MakeValidation(num):  # 一半一半
    data_all = []
    person_list = ReturnPersonPth()
    for path in person_list:
        file_list = []
        for root, dirs, files in os.walk(path):
            # files.sort(key= lambda x:int(x[:-4]))
            for file in files:
                file_list.append(os.path.join(root, file))
        data_all.append(file_list)

    data_1 = []
    data_2 = []
    label = []
    for i in range(num // 2):
        person_same = random.randint(0, 37)
        ill_same = random.randint(0, 63)
        person_same_another = random.randint(0, 37)
        left = data_all[person_same][ill_same]
        right = data_all[person_same_another][ill_same]

        img_1 = Image.open(left)
        img_2 = Image.open(right)
        img_1 = np.array(img_1)[:, :, np.newaxis] / 255.0
        img_2 = np.array(img_2)[:, :, np.newaxis] / 255.0
        data_1.append(img_1)
        data_2.append(img_2)
        label.append(0)
    sampled_list = [n for n in range(64)]
    for i in range(num // 2, num):
        person_diff = random.randint(0, 37)
        ill_diff = random.sample(sampled_list, 2)
        person_diff_another = random.randint(0, 37)
        left = data_all[person_diff][ill_diff[0]]
        right = data_all[person_diff_another][ill_diff[1]]
        img_1 = Image.open(left)
        img_2 = Image.open(right)
        img_1 = np.array(img_1)[:, :, np.newaxis] / 255.0
        img_2 = np.array(img_2)[:, :, np.newaxis] / 255.0
        data_1.append(img_1)
        data_2.append(img_2)
        label.append(1)
    return [np.array(data_1), np.array(data_2)], [label, label, label]

def MakeValidationmultichannel(num):  # 一半一半
    data_all = []
    person_list = ReturnPersonPth()
    for path in person_list:
        file_list = []
        for root, dirs, files in os.walk(path):
            # files.sort(key= lambda x:int(x[:-4]))
            for file in files:
                file_list.append(os.path.join(root, file))
        data_all.append(file_list)

    data_1 = []
    data_2 = []
    label = []
    for i in range(num // 2):
        person_same = random.randint(0, 37)
        ill_same = random.randint(0, 63)
        person_same_another = random.randint(0, 37)
        left = data_all[person_same][ill_same]
        right = data_all[person_same_another][ill_same]

        img_1 = Image.open(left)
        img_2 = Image.open(right)
        img_1 = np.array(img_1)[:, :, np.newaxis] / 255.0
        img_2 = np.array(img_2)[:, :, np.newaxis] / 255.0
        data_1.append(np.tile(img_1,(1,1,3)))
        data_2.append(np.tile(img_2,(1,1,3)))
        label.append(0)
    sampled_list = [n for n in range(64)]
    for i in range(num // 2, num):
        person_diff = random.randint(0, 37)
        ill_diff = random.sample(sampled_list, 2)
        person_diff_another = random.randint(0, 37)
        left = data_all[person_diff][ill_diff[0]]
        right = data_all[person_diff_another][ill_diff[1]]
        img_1 = Image.open(left)
        img_2 = Image.open(right)
        img_1 = np.array(img_1)[:, :, np.newaxis] / 255.0
        img_2 = np.array(img_2)[:, :, np.newaxis] / 255.0
        data_1.append(np.tile(img_1,(1,1,3)))
        data_2.append(np.tile(img_2,(1,1,3)))
        label.append(1)
    return [np.array(data_1), np.array(data_2)], label

def MakeValidationSame(num):  # 全部同类光照的
    data_all = []
    person_list = ReturnPersonPth()
    for path in person_list:
        file_list = []
        for root, dirs, files in os.walk(path):
            # files.sort(key= lambda x:int(x[:-4]))
            for file in files:
                file_list.append(os.path.join(root, file))
        data_all.append(file_list)

    data_1 = []
    data_2 = []
    label = []
    for i in range(num):
        person_same = random.randint(0, 37)
        ill_same = random.randint(0, 63)
        person_same_another = random.randint(0, 37)
        left = data_all[person_same][ill_same]
        right = data_all[person_same_another][ill_same]
        img_1 = Image.open(left)
        img_2 = Image.open(right)
        img_1 = np.array(img_1)[:, :, np.newaxis] / 255.0
        img_2 = np.array(img_2)[:, :, np.newaxis] / 255.0
        data_1.append(img_1)
        data_2.append(img_2)
        label.append(0)
    return [np.array(data_1), np.array(data_2)], [label, label, label]


def MakeValidationDiff(num):  # 全部异类光照的
    data_all = []
    person_list = ReturnPersonPth()
    for path in person_list:
        file_list = []
        for root, dirs, files in os.walk(path):
            # files.sort(key= lambda x:int(x[:-4]))
            for file in files:
                file_list.append(os.path.join(root, file))
        data_all.append(file_list)

    data_1 = []
    data_2 = []
    label = []
    sampled_list = [n for n in range(64)]
    for i in range(num):
        person_diff = random.randint(0, 37)
        ill_diff = random.sample(sampled_list, 2)
        person_diff_another = random.randint(0, 37)
        left = data_all[person_diff][ill_diff[0]]
        right = data_all[person_diff_another][ill_diff[1]]
        img_1 = Image.open(left)
        img_2 = Image.open(right)
        img_1 = np.array(img_1)[:, :, np.newaxis] / 255.0
        img_2 = np.array(img_2)[:, :, np.newaxis] / 255.0
        data_1.append(img_1)
        data_2.append(img_2)
        label.append(1)
    return [np.array(data_1), np.array(data_2)], [label, label, label]

#均匀的产生正负样本
def GenRatio(batch_size):
    height = 192
    weigth = 168
    data_all = []
    person_list = ReturnPersonPth()
    for path in person_list:
        file_list = []
        for root, dirs, files in os.walk(path):
            for file in files:
                file_list.append(os.path.join(root, file))
        data_all.append(file_list)

    while True:
        pos_ratio = random.uniform(0.2, 0.8)
        pos_num = int(batch_size * pos_ratio)
        height_ratio = random.uniform(0.8, 1.2)
        weigth_ratio = random.uniform(0.9, 1.5)
        data_1 = []
        data_2 = []
        label = []
        for i in range(pos_num):
            person_same = random.randint(0, 37)
            ill_same = random.randint(0, 63)
            person_same_another = random.randint(0, 37)
            left = data_all[person_same][ill_same]
            right = data_all[person_same_another][ill_same]

            img_1 = Image.open(left)
            img_2 = Image.open(right)
            img_1 = np.array(img_1)
            img_2 = np.array(img_2)
            img_1 = cv2.resize(img_1, (int(weigth * weigth_ratio), int(height * height_ratio)))
            img_2 = cv2.resize(img_2, (int(weigth * weigth_ratio), int(height * height_ratio)))
            img_1 = img_1[:, :, np.newaxis] / 255.0
            img_2 = img_2[:, :, np.newaxis] / 255.0
            data_1.append(img_1)
            data_2.append(img_2)
            label.append(0)
        sampled_list = [n for n in range(64)]
        for i in range(pos_num, batch_size):
            person_diff = random.randint(0, 37)
            ill_diff = random.sample(sampled_list, 2)
            person_diff_another = random.randint(0, 37)
            left = data_all[person_diff][ill_diff[0]]
            right = data_all[person_diff_another][ill_diff[1]]
            img_1 = Image.open(left)
            img_2 = Image.open(right)
            img_1 = np.array(img_1)
            img_2 = np.array(img_2)
            img_1 = cv2.resize(img_1, (int(weigth * weigth_ratio), int(height * height_ratio)))
            img_2 = cv2.resize(img_2, (int(weigth * weigth_ratio), int(height * height_ratio)))
            img_1 = img_1[:, :, np.newaxis] / 255.0
            img_2 = img_2[:, :, np.newaxis] / 255.0
            data_1.append(img_1)
            data_2.append(img_2)
            label.append(1)
        yield [np.array(data_1), np.array(data_2)], [np.array(label), np.array(label), np.array(label)]


def GenRatiomultichannel(batch_size):
    height = 192
    weigth = 168
    data_all = []
    person_list = ReturnPersonPth()
    for path in person_list:
        file_list = []
        for root, dirs, files in os.walk(path):
            for file in files:
                file_list.append(os.path.join(root, file))
        data_all.append(file_list)

    while True:
        pos_ratio = random.uniform(0.1, 0.9)
        pos_num = int(batch_size * pos_ratio)
        height_ratio = random.uniform(0.8, 1.2)
        weigth_ratio = random.uniform(0.85, 1.25)
        data_1 = []
        data_2 = []
        label = []
        for i in range(pos_num):
            person_same = random.randint(0, 37)
            ill_same = random.randint(0, 63)
            person_same_another = random.randint(0, 37)
            left = data_all[person_same][ill_same]
            right = data_all[person_same_another][ill_same]

            img_1 = Image.open(left)
            img_2 = Image.open(right)
            img_1 = np.array(img_1)
            img_2 = np.array(img_2)
            img_1 = cv2.resize(img_1, (int(weigth * weigth_ratio), int(height * height_ratio)))
            img_2 = cv2.resize(img_2, (int(weigth * weigth_ratio), int(height * height_ratio)))
            # img_1 = cv2.resize(img_1, (224,224))
            # img_2 = cv2.resize(img_2, (224,224))

            img_1 = img_1[:, :, np.newaxis] / 255.0
            img_2 = img_2[:, :, np.newaxis] / 255.0
            data_1.append(np.tile(img_1,(1,1,3)))
            data_2.append(np.tile(img_2,(1,1,3)))
            label.append(0)
        sampled_list = [n for n in range(64)]
        for i in range(pos_num, batch_size):
            person_diff = random.randint(0, 37)
            ill_diff = random.sample(sampled_list, 2)
            person_diff_another = random.randint(0, 37)
            left = data_all[person_diff][ill_diff[0]]
            right = data_all[person_diff_another][ill_diff[1]]
            img_1 = Image.open(left)
            img_2 = Image.open(right)
            img_1 = np.array(img_1)
            img_2 = np.array(img_2)
            img_1 = cv2.resize(img_1, (int(weigth * weigth_ratio), int(height * height_ratio)))
            img_2 = cv2.resize(img_2, (int(weigth * weigth_ratio), int(height * height_ratio)))
            # img_1 = cv2.resize(img_1, (224,224))
            # img_2 = cv2.resize(img_2, (224,224))
            img_1 = img_1[:, :, np.newaxis] / 255.0
            img_2 = img_2[:, :, np.newaxis] / 255.0
            data_1.append(np.tile(img_1,(1,1,3)))
            data_2.append(np.tile(img_2,(1,1,3)))
            label.append(1)
        yield [np.array(data_1), np.array(data_2)], np.array(label)


def Gen_64(batch_size):
    def myGenerator(batch_size):
        height = 192
        weigth = 168
        path_list = ReturnData_Label()
        num = len(path_list)
        while True:
            pos_ratio = random.uniform(0, 0.3)
            pos_num = int(batch_size * pos_ratio)
            nege_num = batch_size - pos_num
            height_ratio = random.uniform(0.8, 1.2)
            weigth_ratio = random.uniform(0.9, 1.5)
            data_1 = []
            data_2 = []
            label = []
            for i in range(batch_size):
                k = random.randint(0, num - 1)
                j = random.randint(0, num - 1)
                left = path_list[k]
                right = path_list[j]
                img_1 = Image.open(left)
                img_2 = Image.open(right)
                img_1 = np.array(img_1)
                img_2 = np.array(img_2)
                img_1 = cv2.resize(img_1, (int(weigth * weigth_ratio), int(height * height_ratio)))
                img_2 = cv2.resize(img_2, (int(weigth * weigth_ratio), int(height * height_ratio)))
                img_1 = img_1[:, :, np.newaxis] / 255.0
                img_2 = img_2[:, :, np.newaxis] / 255.0
                data_1.append(img_1)
                data_2.append(img_2)

                # if left.split('\\')[4] == right.split('\\')[4]:
                #     label_1[i] = 1
                if left.split('\\')[-1][10:] == right.split('\\')[-1][10:]:
                    label.append(0)
                else:
                    label.append(1)
            # yield [data_1,data_2],[label_1,label_3,label_2]
            yield [np.array(data_1), np.array(data_2)], np.array(label)

def GenSeq(batch_size):
    path = r"E:\01DeepFakesDetection\04FF++_Full_Videos\Images\DF_crop224\train"
    A, B = os.listdir(path)
    path_ = [path + '\\' + A, path + '\\' + B]
    sub_ = [os.listdir(path_[0]), os.listdir(path_[1])]

    while True:
        data = []
        label = []
        for i in range(batch_size):
            _class_left = random.randint(0, 1)
            _num = random.randint(0, 718)
            image_path = os.listdir(path_[_class_left] + '\\' + sub_[_class_left][_num])
            # image_path.sort(key=lambda x: int(x[:-4]))
            len_ = len(image_path)
            start = random.randint(0, len_ - 1)
            img = cv2.imread(path_[_class_left] + '\\' + sub_[_class_left][_num] + '\\' + image_path[start]) / 255.0
            # img = cv2.resize(img,(224,224))
            data.append(img)
            if _class_left==1:
                label.append(1)
            else:
                label.append(0)
        yield np.array(data), np.array(label)

def GenMask(batch_size):
    path = r"E:\01DeepFakesDetection\04FF++_Full_Videos\BigImagesWithMask\F2F\train"
    A, B,C = os.listdir(path)
    path_ = [path + '\\' + A, path + '\\' + B,path + '\\' + C]
    sub_ = [os.listdir(path_[0]), os.listdir(path_[1]),os.listdir(path_[2])]

    while True:
        data = []
        label = []
        mask = []
        for i in range(batch_size):
            _class_ = random.randint(0, 1)
            _num = random.randint(0, 719)
            image_path = os.listdir(path_[_class_] + '\\' + sub_[_class_][_num])
            # image_path.sort(key=lambda x: int(x[:-4]))
            len_ = len(image_path)
            start = random.randint(0, len_ - 1)
            img = cv2.imread(path_[_class_] + '\\' + sub_[_class_][_num] + '\\' + image_path[start]) / 255.0
            img = cv2.resize(img,(224,224))
            data.append(img)

            if _class_ == 0:
                msk = cv2.imread(path_[2] + '\\' + sub_[2][_num] + '\\' + image_path[start],0)
                msk = cv2.resize(msk,(224,224))
                msk = msk>8
                mask.append(msk[:,:,np.newaxis])
            else:
                mask.append(np.zeros((224,224,1)))

            if _class_==1:
                label.append(1)
            else:
                label.append(0)
        yield np.array(data), [np.array(label),np.array(mask)]

def GenMaskVal(batch_size):
    path = r"E:\01DeepFakesDetection\04FF++_Full_Videos\BigImagesWithMask\NT\val"
    A, B,C = os.listdir(path)
    path_ = [path + '\\' + A, path + '\\' + B,path + '\\' + C]
    sub_ = [os.listdir(path_[0]), os.listdir(path_[1]),os.listdir(path_[2])]

    while True:
        data = []
        label = []
        mask = []
        for i in range(batch_size):
            _class_ = random.randint(0, 1)
            _num = random.randint(0, 139)
            image_path = os.listdir(path_[_class_] + '\\' + sub_[_class_][_num])
            # image_path.sort(key=lambda x: int(x[:-4]))
            len_ = len(image_path)
            start = random.randint(0, len_ - 1)
            img = cv2.imread(path_[_class_] + '\\' + sub_[_class_][_num] + '\\' + image_path[start]) / 255.0
            img = cv2.resize(img,(224,224))
            data.append(img)

            if _class_ == 0:
                msk = cv2.imread(path_[2] + '\\' + sub_[2][_num] + '\\' + image_path[start],0)
                msk = cv2.resize(msk,(224,224))
                msk = msk>8
                mask.append(msk[:,:,np.newaxis])
            else:
                mask.append(np.zeros((224,224,1)))

            if _class_==1:
                label.append(1)
            else:
                label.append(0)
        yield np.array(data), [np.array(label),np.array(mask)]

def GenMaskdata(num):
    path = r"E:\01DeepFakesDetection\04FF++_Full_Videos\BigImagesWithMask\F2F\val"
    A, B,C = os.listdir(path)
    path_ = [path + '\\' + A, path + '\\' + B,path + '\\' + C]
    sub_ = [os.listdir(path_[0]), os.listdir(path_[1]),os.listdir(path_[2])]

    data = []
    label = []
    mask = []
    for i in range(num):
        _class_ = random.randint(0, 1)
        _num = random.randint(0, 139)
        image_path = os.listdir(path_[_class_] + '\\' + sub_[_class_][_num])
        # image_path.sort(key=lambda x: int(x[:-4]))
        len_ = len(image_path)
        start = random.randint(0, len_ - 1)
        img = cv2.imread(path_[_class_] + '\\' + sub_[_class_][_num] + '\\' + image_path[start]) / 255.0
        img = cv2.resize(img,(224,224))
        data.append(img)

        if _class_ == 0:
            msk = cv2.imread(path_[2] + '\\' + sub_[2][_num] + '\\' + image_path[start],0)
            print(path_[2] + '\\' + sub_[2][_num] + '\\' + image_path[start])
            msk = cv2.resize(msk,(224,224))
            msk = msk>8
            mask.append(msk[:,:,np.newaxis])
        else:
            mask.append(np.zeros((224,224,1)))

        if _class_==1:
            label.append(1)
        else:
            label.append(0)
    return np.array(data), [np.array(label),np.array(mask)]

def _GenVal(num):
    path = r"E:\01DeepFakesDetection\04FF++_Full_Videos\Images\DF_crop224\test"
    A, B = os.listdir(path)
    path_ = [path + '\\' + A, path + '\\' + B]
    sub_ = [os.listdir(path_[0]), os.listdir(path_[1])]

    while True:
        data = []
        label = []
        for i in range(num):
            _class_left = random.randint(0, 1)
            _num = random.randint(0, 139)
            image_path = os.listdir(path_[_class_left] + '\\' + sub_[_class_left][_num])
            # image_path.sort(key=lambda x: int(x[:-4]))
            len_ = len(image_path)
            start = random.randint(0, len_ - 1)
            img = cv2.imread(path_[_class_left] + '\\' + sub_[_class_left][_num] + '\\' + image_path[start]) / 255.0
            # img = cv2.resize(img,(224,224))
            data.append(img)
            if _class_left==1:
                label.append(1)
            else:
                label.append(0)
        yield np.array(data), np.array(label)



def motion_valid(size):
    recon_loss = np.zeros((size, 224, 224, 3))
    apper_loss = np.zeros((size, 56, 56, 32))

    path = r"E:\DATA\Deepfake\F2F_crop224\val"
    A, B = os.listdir(path)
    path_ = [path + '\\' + A, path + '\\' + B]
    sub_ = [os.listdir(path_[0]), os.listdir(path_[1])]
    data_1 = []
    data_2 = []
    for i in range(size):
        _class = random.randint(0, 1)
        _num = random.randint(0, 139)
        image_path = os.listdir(path_[_class] + '\\' + sub_[_class][_num])
        image_path.sort(key=lambda x: int(x[:-4]))
        len_ = len(image_path)
        start_ = random.randint(0, len_ - 2)
        img_1 = cv2.imread(path_[_class] + '\\' + sub_[_class][_num] + '\\' + image_path[start_]) / 255.0
        data_1.append(img_1)
        img_2 = cv2.imread(path_[_class] + '\\' + sub_[_class][_num] + '\\' + image_path[start_ + 1]) / 255.0
        data_2.append(img_2)

    return [np.array(data_1), np.array(data_2)], [recon_loss, apper_loss]


def FinalDataGen(time_step):
    path = r"E:\DATA\Deepfake\F2F_crop224\train"
    A, B = os.listdir(path)
    path_ = [path + '\\' + A, path + '\\' + B]
    sub_ = [os.listdir(path_[0]), os.listdir(path_[1])]

    while True:
        data = []
        label = []
        _class = random.randint(0, 1)
        if _class == 1:
            label.append(1)
        else:
            label.append(0)
        _num = random.randint(0, 719)
        image_path = os.listdir(path_[_class] + '\\' + sub_[_class][_num])
        image_path.sort(key=lambda x: int(x[:-4]))
        len_ = len(image_path)
        start_ = random.randint(0, len_ - time_step - 1)
        for i in range(time_step):
            time_sample = np.zeros((224, 224, 8))
            img_ill_1 = cv2.imread(path_[_class] + '\\' + sub_[_class][_num] + '\\' + image_path[start_+i], 0) / 255.0
            img_ill_2 = cv2.imread(path_[_class] + '\\' + sub_[_class][_num] + '\\' + image_path[start_ +i+ 1],0) / 255.0
            img_motion_1 = cv2.imread(path_[_class] + '\\' + sub_[_class][_num] + '\\' + image_path[start_+i]) / 255.0
            img_motion_2 = cv2.imread(path_[_class] + '\\' + sub_[_class][_num] + '\\' + image_path[start_ +i+ 1]) / 255.0
            time_sample[:, :, 0] = img_ill_1[:, :]
            time_sample[:, :, 1] = img_ill_2[:, :]
            time_sample[:, :, 2:5] = img_motion_1[:, :, :]
            time_sample[:, :, 5:8] = img_motion_2[:, :, :]
            data.append(time_sample)
        yield np.array(data), np.array(label)

def myGeneratorFrame(pth,batch_size):
    print("Generator Initializing")
    path_list = ReadPath_shuffle(pth,0)
    video_num = len(path_list)

    frame_list = []
    for index in range(video_num):
        video_frames = ReadPic_shuffle(path_list[index])
        frame_list+=video_frames
    random.shuffle(frame_list)
    length = len(frame_list)
    start_index = 0
    print("Generator Loaded")
    while True:
        # data = np.zeros((batch_size, frame_length, 224, 224, 3))
        data = []
        label = []

        end_index = start_index+batch_size
        picked_frames = frame_list[start_index:end_index]
        start_index = end_index

        if start_index+batch_size>length:
             random.shuffle(frame_list)
             start_index = 0

        for (i,frame) in enumerate(picked_frames):
            if "original" in frame:
                label.append(1)
            else:
                label.append(0)
            data.append(cv2.resize(cv2.imread(picked_frames[i]),(224,224))/255.0)
        yield np.array(data),np.array(label)

if __name__ == "__main__":
    a = FinalDataGen(8)
    print(a.__next__()[1].shape)
