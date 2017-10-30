#encoding:UTF-8

import os
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
from sklearn.svm import NuSVC, SVC
import datetime
import pickle


#calculate the area
def area(p):
    p = p.reshape((-1, 2))
    return 0.5 * abs(sum(x0*y1 - x1*y0 
                    for ((x0, y0), (x1, y1)) in segments(p)))
def segments(p):
    return zip(p, np.concatenate((p[1:], [p[0]])))

def calc_xy(p0, p1, p2):
    cos = calc_cos(p0, p1, p2)
    dis = calc_dis(p0, p2)
    return dis * cos, dis * np.sqrt(1 - np.square(cos))

def calc_dis(p0, p1):
    return np.sqrt(np.sum(np.square(p0-p1)))

def calc_cos(p0, p1, p2):
    A = p1 - p0
    B = p2 - p0
    num = np.dot(A, B)
    demon = np.linalg.norm(A) * np.linalg.norm(B)
    return num / demon

def calc_new_xy(boxes):
    box0 = boxes[:8]
    box1 = boxes[8:]
    x, y = calc_xy(box1[4:6], box1[6:], box0[:2])
    dis = calc_dis(box1[4:6], box1[6:])
    area0 = area(box0)
    area1 = area(box1)
    return x/dis, y/dis, area0/area1

if __name__ == '__main__':
    test = True
    path = '/media/zhaoke/b0685ee4-63e3-4691-ae02-feceacff6996/data/'
    paths = os.listdir(path)
    paths = [i for i in paths if '.txt' in i]
    boxes = np.empty((8*len(paths), 9))
    cnt = 0
    for txt in paths:
        f = open(path+txt, 'r')
        lines = f.readlines()
        f.close()
        lines = [i.replace('\n', '').split(',') for i in lines]
        lines = np.array(lines).astype(np.uint32)
        lines = lines[lines[:, -1]<=8]
        boxes[cnt*8:cnt*8+8] = lines
        cnt += 1
    idboxes = boxes[boxes[:, 8]==7]
    idboxes = np.tile(idboxes[:, :8], (1, 8))
    idboxes = idboxes.reshape((-1, 8))
    boxes_idboxes = np.concatenate((boxes[:, :8], idboxes), axis=1)
    start_time = datetime.datetime.now()
    print start_time
    new_xy = np.apply_along_axis(calc_new_xy, 1, boxes_idboxes)
    end_time = datetime.datetime.now()
    print end_time - start_time
    if test:
        with open('clf_address_v1.pickle', 'rb') as f:
            clf = pickle.load(f)
        cnt = 0
        for i, xy in enumerate(new_xy):
            cls = int(clf.predict([xy])[0])
            if cls == int(boxes[i, 8]):
                cnt += 1
            if i % 10000 == 0 and i != 0:
                print i, ':', float(cnt) / i
    else:
        clf = SVC()
        start_time = datetime.datetime.now()
        print start_time
        clf.fit(new_xy[:], boxes[:, 8])
        end_time = datetime.datetime.now()
        print end_time - start_time
        with open('clf.pickle', 'wb') as f:
            pickle.dump(clf, f)
        print 'end'