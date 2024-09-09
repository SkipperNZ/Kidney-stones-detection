import os
from ultralytics import YOLO
import argparse



def main(args):
    model = YOLO(args.weights)  # Загружаем модель с указанными весами
  
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        project=args.project,
        name=args.name,
        device=args.device
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov10s.pt', help='modeel to load or path to weights')
    parser.add_argument('--data', type=str, default='./dataset.yaml', help='Config file')
    parser.add_argument('--epochs', type=int, default=50, help='epochs num')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--img_size', type=int, default=640, help='default img size')
    parser.add_argument('--project', type=str, default='runs/train', help='path to save results')
    parser.add_argument('--name', type=str, default='exp', help='exp name')
    parser.add_argument('--device', type=str, default='0', help='devive num or cpu')

    args = parser.parse_args()
    main(args)
