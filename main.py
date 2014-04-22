import numpy as np
import csv
import sys

import skimage.io as io
from skimage.filter import threshold_otsu
from skimage.color import rgb2gray
from skimage.morphology import label
from skimage.measure import regionprops


def cost(im, et):
    '''
    Measure between image and etalon based on hamming distance
    '''
    shape = min(im.shape[0], et.shape[0]), min(im.shape[1], et.shape[1])
    out = ((im.shape[0] + et.shape[0] - 2 * shape[0]) * 
        (im.shape[1] + et.shape[1] - 2 * shape[1]))
    return (sum(sum(np.logical_xor(im[:shape[0], :shape[1]], 
        et[:shape[0], :shape[1]]))) + out + 0.0) / (im.shape[0] * im.shape[1])


def get_d(image_file):
    '''
    Extracting etalons from image
    '''
    max_cost = 0.18
    max_et = 100
    max_area = 30
    
    image = rgb2gray(io.imread(image_file))
    tresh = threshold_otsu(image)
    bi = image < tresh
    li = label(bi)
    dt = []
    for region in regionprops(li, ['Area', 'BoundingBox']):
        if region['Area'] < max_area:
            continue
        minr, minc, maxr, maxc = region['BoundingBox']
        imr = bi[minr:maxr, minc:maxc]
        h = False
        for t in dt:
            if cost(imr, t) < max_cost:
                h = True
                break
        if h:
            continue
        else:
            dt.append(imr)
        if len(dt) > max_et:
            break
    return dt


def load(info_file = 'info.txt'):
    test = []
    train = []
    with open(info_file, 'rb') as f:
        fr = csv.reader(f, delimiter=';')
        for row in fr:
            print row[0]
            d = dict()
            if len(row) == 1:
                d['file'] = row[0]
                d['font'] = None
                d['weight'] = None
                d['style'] = None
                d['data'] = get_d(row[0])
                test.append(d)
            else:
                d['file'] = row[0]
                d['font'] = row[1]
                d['weight'] = row[2]
                d['style'] = row[3]
                d['data'] = get_d(row[0])
                train.append(d)
            print len(d['data'])
    return train, test


def classify(train, test):
    for d in test:
        min_cost = 100
        min_t = None
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
        d['font'] = min_t['font']
        d['weight'] = min_t['weight']
        d['style'] = min_t['style']
        print d['file'], ':'
        print d['font'], ';', d['weight'], ';', d['style']
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
