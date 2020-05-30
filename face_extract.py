import json
import os,time
import cv2
import sys
from mtcnn import MTCNN
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=config)
ktf.set_session(session)
# {'label': 'FAKE', 'split': 'train', 'original': 'llfxvxzllg.mp4'}

# 0,10,19,20,28,3,1,29,11,2

detector = MTCNN()

def extractframes(path,dst):
    capture=cv2.VideoCapture(path)
    cap=capture.get(cv2.CAP_PROP_FRAME_COUNT)
    i=0
    count=0
    filename=path.split('/')[-1].split('.')[0]
    if not os.path.exists(dst+'_0'):
        os.makedirs(dst+'_0')
    pos=None
    valcount=0
    while True:
        ret, frame = capture.read()
        if ret is False:
            break
        face,pos_=detectface(frame,pos)
        pos=pos_
        if face is not None:
            cv2.imwrite(dst+'_'+str(i)+'/'+filename+"_"+str(count)+".jpg",face)
            count=count+1
            valcount+=1
        if count==40:
            pos=None
            count=0
            i+=1
            if not os.path.exists(dst+'_'+str(i)):
                os.makedirs(dst+'_'+str(i))
    print(path,cap,count)
    capture.release()
    return valcount,cap

def detectface(img,pos):
    if pos==None:
        img_height=img.shape[0]
        img_width=img.shape[1]
        # 人脸数rects
        rects = detector.detect_faces(img)
        max_area=0
        startx=None
        for i in range(len(rects)):
            d=rects[i]
            r=d['box']
            if d['confidence']<0.85:
                pass
            else:
                height = r[2]
                width = r[3]
                area=height*width
                maxlen=int(max(height,width)*1.4)
                if area > max_area:
                    max_area=area
                    centerx=int(r[1]+r[3]/2)
                    centery=int(r[0]+r[2]/2)
                    startx=centerx-maxlen//2
                    starty=centery-maxlen//2
                    if startx<0:
                        startx=0
                    elif startx+maxlen>img_height:
                        startx=img_height-maxlen
                    if starty<0:
                        starty=0
                    elif starty+maxlen>img_width:
                        starty=img_width-maxlen
        if startx is not None:
            pos=[startx,starty,maxlen]
            return img[startx:startx+maxlen,starty:starty+maxlen,:],pos
        else:
            return None,None
    else:
        x,y,maxlen=pos
        return img[x:x+maxlen,y:y+maxlen,:],pos

datadir = '/Data/olddata_E/01DeepFakesDetection/04FF++_Full_Videos/Videos/Real/'
imgdir = '/Data/data1/04FF/Real/'
data=[]
count=0
totalframes=0
for c in os.listdir(datadir):
    cpath=datadir+c
    # for d in os.listdir(cpath):
    #     dpath=cpath+'/'+d
    for video in os.listdir(cpath):
        if video == '@eaDir' or video == 'Thumbs.db':
            pass
        else:
            videopath=cpath+'/'+video
            data.append(videopath)

data.sort()
# print(len(data))
i=0
totalcount=0
totalframes=0
num=873
for video in data:
    start=time.time()
    videopath=video
    splits=videopath.split('/')
    imgpath=imgdir+splits[6]+'/'+splits[7]+'/'+splits[8][:-4]
    print(videopath)
    try:
        cap,count=extractframes(videopath,imgpath)
    except:
        print('pass')
    totalcount+=count
    totalframes+=cap
    print(i,imgpath,"used time:",time.time()-start)
    i+=1

print((totalcount-totalframes)/totalframes)