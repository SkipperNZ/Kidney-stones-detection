import cv2
from PIL import Image
from pathlib import Path
import numpy as np

LABELS =  ['Oxalate', 'Phosphate', 'Urate', 'Stone']


def predict_pic(path_to_pic, model):
    predict = model.predict(path_to_pic)
    print(f"Picture name: {path_to_pic.name}")

    for i, pred in enumerate(predict[0].boxes.cls):
        print(f"Predict: {LABELS[int(pred)]}, Confidence: {predict[0].boxes.conf[i]}")
    
    return predict



def predict_and_plot_picture(path_to_pic, model):
    predict = predict_pic(path_to_pic, model)
    return Image.fromarray(predict[0].plot()[..., ::-1])
    

def plot_gt(picture, annotations, labels):
    for label, bbox in annotations:

        x_center, y_center, width, height = bbox
        h, w, _ = picture.shape
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        

        cv2.rectangle(picture, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=1)
        cv2.putText(picture, f'Gt: {labels[int(label)]}', (x1, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 1)

    return picture



def predict_from_dataset(path_to_pic, model):
    path_to_anotations = str(path_to_pic).replace('.png', '.txt').replace("images", "labels")

    with open(path_to_anotations, "r") as f:
        annotations = list(map(float, f.read().split()))
        annotations = [(annotations[i], annotations[i+1:i+5]) for i in range(0, len(annotations), 5)]
        
        for box in annotations:
            print(f"Gt:{LABELS[int(box[0])]}")
        
            

    predict = predict_pic(path_to_pic, model)
    picture = predict[0].plot()
    picture = plot_gt(picture, annotations, LABELS)

    picture = Image.fromarray(picture[..., ::-1])


    return picture