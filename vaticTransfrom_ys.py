import os
import numpy as np
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-d',dest='dataset')
parser.add_argument('-m',dest='model')
parser.add_argument('-t',dest='threshold',default=50)

args=parser.parse_args()

mapper = {"bike":"bicycle", 'scooter': "motorcycle", "trailer-head": "truck",
                  "van": "car", "pickup": "car", "sedan/suv": "car",
                  "suitcase": "__background__", "luggage": "__background__",
                  "backpack": "__background__", "handbag": "__background__"}


def bboxTransform(bbox):
    bbox[2]=bbox[0]+bbox[2]
    bbox[3]=bbox[1]+bbox[3]
    return bbox


def transform(dataName,modelName,threshold):
    fileName = 'res'
    directory = '/root/data/data-{}/{}/{}/set01/V000.txt'.format(dataName,fileName,modelName)
    print directory
    f = open(directory,'r')
    preds = {}
    cnt = 1
    fileName = 'transformed'
    directory = '/root/data/data-{}/{}/{}/set01'.format(dataName,fileName,modelName)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for lines in f:
        placeHolder = str(lines).split(",")
        score = float(placeHolder[5])
        frameNum = placeHolder[0]
        with open('/root/data/data-{}/{}/{}/set01/set01_V000_I{}.jpg.txt'.format(dataName,fileName,modelName,frameNum),'ab') as w:
            if score <= threshold:
                continue
            cnt = int(frameNum)
	    pred = []
            pred.append(placeHolder[6].split('\n')[0])#label
            pred.append(str(score/100.0))
            bbox = [float(i) for i in placeHolder[1:5]]
            bbox = bboxTransform(bbox)
            pred = pred + [str(i) for i in bbox]
            pred[5] = pred[5]+'\n'
            annotations =  " ".join(pred)
            w.write(annotations)
    print cnt

    f.close()

def processGT(dataName):
    jsonInput = open('/root/data/data-{}/annotations.json'.format(dataName),'r')

    annos = json.load(jsonInput)

    frames = {}

    for key,value in annos['1'].iteritems():
        directory = '/root/data/data-{}/ground-truth/set01'.format(dataName)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if key == '0':
            continue
        w = open('/root/data/data-{}/ground-truth/set01/set01_V000_I{}.jpg.txt'.format(dataName,key),'w')
        for key, annos in value.iteritems():
            line = []
            label = annos['label']
            if label.lower() in mapper:
                label = mapper[label.lower()]
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
threshold = int(args.threshold)

processGT(dataName)
transform(dataName,modelName,threshold)

