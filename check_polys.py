#encoding

import cv2
import numpy as np
import os

path = '/home/zhaoke/justrypython/EAST/data/data/'

def polygon_area(poly):
    '''
    compute area of a polygon
    :param poly:
    :return:
    '''
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge)/2.

def check_and_validate_polys():
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    paths = os.listdir(path)
    errs = []
    for i in paths:
        if '.txt' not in i:
            continue
        f = open(path+i, 'r')
        lines = f.readlines()
        polys =  np.array([j.split(',') for j in lines])[:, :8]
        polys = polys.astype('int32')
        polys = polys.reshape([-1, 4, 2])
        for poly in polys:
            p_area = polygon_area(poly)
            if abs(p_area) < 1:
                # print poly
                errs.append(i)
                break
            elif p_area > 0:
                errs.append(i)
                break
    for i in sorted(errs):
        print i
        os.system('rm %s'%(path+i))
        print 'rm %s'%(path+i)
        os.system('rm %s'%(path+i[:-4]+'.jpg'))
        print 'rm %s'%(path+i[:-4]+'.jpg')
            
if __name__ == '__main__':
    check_and_validate_polys()