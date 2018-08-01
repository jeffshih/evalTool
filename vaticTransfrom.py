import os
import numpy as np
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-d',dest='dataset')
parser.add_argument('-m',dest='model')
parser.add_argument('-t',dest='threshold',default=50)

args=parser.parse_args()


def bboxTransform(bbox):
    bbox[2]=bbox[0]+bbox[2]
    bbox[3]=bbox[1]+bbox[3]
    return bbox


def transform(dataName,modelName,threshold):
    fileName = 'res'
    directory = 'data-{}/{}/{}/set01/V000.txt'.format(dataName,fileName,modelName)
    f = open(directory,'r')
    preds = {}
    prev = 1
    fileName = 'transformed'
    directory = './data-{}/{}/{}/set01'.format(dataName,fileName,modelName)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for lines in f:
        placeHolder = str(lines).split(",")
        score = float(placeHolder[5])
        frameNum = placeHolder[0]
        with open('data-{}/{}/{}/set01/set01_V000_I{}.jpg.txt'.format(dataName,fileName,modelName,frameNum),'ab') as w:
            if score <= threshold:
                continue
            pred = []
            pred.append(placeHolder[6].split('\n')[0])#label
            pred.append(str(score/100.0))
            bbox = [float(i) for i in placeHolder[1:5]]
            bbox = bboxTransform(bbox)
            pred = pred + [str(i) for i in bbox]
            pred[5] = pred[5]+'\n'
            annotations =  " ".join(pred)
            w.write(annotations)
            cnt = int(frameNum)
    print cnt

    f.close()

def processGT(dataName):
    jsonInput = open('data-{}/annotations.json'.format(dataName),'r')

    annos = json.load(jsonInput)

    frames = {}

    for key,value in annos['1'].iteritems():
        directory = 'data-{}/ground-truth/set01'.format(dataName)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if key == '0':
            continue
        w = open('data-{}/ground-truth/set01/set01_V000_I{}.jpg.txt'.format(dataName,key),'w')
        for key, annos in value.iteritems():
            line = []
            label = annos['label']
            line.append(label)
            x1 = annos['x1']
            y1 = annos['y1']
            height = annos['height']
            width = annos['width']
            bbox = [x1,y1,width,height]
            bbox = bboxTransform(bbox)
            line = line + [str(i) for i in bbox]
            line[4] = line[4] + '\n'
            annotations =  " ".join(line)
            w.writelines(annotations)
        w.close()



dataName = args.dataset
modelName = args.model
threshold = args.threshold

processGT(dataName)
transform(dataName,modelName,threshold)

