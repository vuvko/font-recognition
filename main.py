import numpy as np
import csv
import sys
from time import clock

import skimage.io as io
from skimage.filter import threshold_otsu
from skimage.color import rgb2gray
from skimage.morphology import label
from skimage.measure import regionprops
from skimage.transform import AffineTransform, warp, resize


def cost(im, et):
    '''
    Measure between image and etalon based on hamming distance
    '''
    return ((sum(sum(np.logical_xor(im, et))) + 0.0) 
        / (im.shape[0] * im.shape[1]))


def get_d(image):
    '''
    Extracting etalons from image
    '''
    print 'Extracting...'
    max_cost = 0.2
    max_et = 100
    max_area = 30
    max_transform = 3
    
    image = rgb2gray(image)
    tresh = threshold_otsu(image)
    dt = []
    
    for it in xrange(max_transform):
        bi = image < tresh
        li = label(bi)
        print it
        for region in regionprops(li, ['Area', 'BoundingBox']):
            if region['Area'] < max_area:
                continue
            minr, minc, maxr, maxc = region['BoundingBox']
            imr = bi[minr:maxr, minc:maxc]
            imr = resize(imr, (30, 30))
            h = False
            for t in dt:
                if cost(imr, t) < max_cost:
                    h = True
                    break
            if h:
                continue
            else:
                dt.append(imr)
            if len(dt) >= max_et:
                return dt
        if it % 2:
            sx = 0.9 - (it + 0.0) / 10
            sy = 1.0 + (it + 0.0) / 11
            rt = 0.02 + (it + 0.0) * 0.01
        else:
            sx = 0.9 + (it + 0.0) / 10
            sy = 1.0 - (it + 0.0) / 11
            rt = -0.03 - (it + 0.0) * 0.015
        tform = AffineTransform(scale=(sx, sy), rotation=rt)
        image = warp(image, tform)
        print len(dt)
    return dt


def load(info_file = 'info.txt'):
    test = []
    train = []
    with open(info_file, 'rb') as f:
        fr = csv.reader(f, delimiter=';')
        for row in fr:
            print row[0]
            start = clock()
            d = dict()
            if len(row) == 1:
                d['file'] = row[0]
                d['font'] = None
                d['weight'] = None
                d['style'] = None
                d['data'] = get_d(io.imread(row[0]))
                test.append(d)
            else:
                d['file'] = row[0]
                d['font'] = row[1]
                d['weight'] = row[2]
                d['style'] = row[3]
                d['data'] = get_d(io.imread(row[0]))
                train.append(d)
            end = clock()
            print len(d['data'])
            print end - start
    return train, test


def classify(train, test):
    for d in test:
        min_cost = 100
        min_t = None
        start = clock()
        for t in train:
            sc = 0
            for im in d['data']:
                min_c = 100
                for et in t['data']:
                    c = cost(im, et)
                    if c < min_c:
                        min_c = c
                sc = sc + min_c
            if sc < min_cost:
                min_cost = sc
                min_t = t
        end = clock()
        d['font'] = min_t['font']
        d['weight'] = min_t['weight']
        d['style'] = min_t['style']
        print d['file'], ':'
        print d['font'], ';', d['weight'], ';', d['style']
        print end - start
    return test


def save_answer(train, test, result_file='result.txt'):
    with open(result_file, 'wb') as f:
        fw = csv.writer(f, delimiter=';')
        for t in train:
            fw.writerow([t['file'], t['font'], t['weight'], t['style']])
        for t in test:
            fw.writerow([t['file'], t['font'], t['weight'], t['style']])


def run(info_file='info.txt', result_file='result.txt'):
    print 'Loading data from', info_file
    train, test = load(info_file)
    print 'Classifying...'
    test = classify(train, test)
    print 'Saving results in', result_file
    save_answer(train, test, result_file)
    print 'Done.'

if __name__ == '__main__':
    if len(sys.argv) > 2:
        result_file = sys.argv[2]
    else:
        result_file = 'result.txt'
    if len(sys.argv) > 1:
        info_file = sys.argv[1]
    else:
        info_file = 'info.txt'

    run(info_file, result_file)
