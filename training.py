import glob
from ultralytics import YOLO
import torch

from config import YOLO_DATASET

if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #model = YOLO("../runs/detect/train/weights/best.pt")
    model = YOLO("yolov8m.yaml")
    results = model.train(
        data=f"{YOLO_DATASET}/dataset.yaml",  
        device=device,
        batch=0.9,
        patience=20,
        save=True, 
        resume=False,
        val=True,
        plots=True,
        imgsz = 640,
        save_period=10,
        # dfl = 4.5,
        # box = 7.5,
        epochs=100,   
        cls=0.8,
        augment=True,
        hsv_s=0.7,
        hsv_v=0.4,
        hsv_h=0.2,        
        dropout=0.1,
        perspective = 0.001,
        degrees = 0.3,
        scale = 0.1,
        mixup = 0.0,
        mosaic = 1.0,
        copy_paste = 1.0,        
        shear = 0.0,
        bgr = 0.0,
    )

    
    results = model.val()
    
    success = model.export(path="yolov8mTruco.pt")

    pictures = glob.glob("./datasets/custom/*")
    print (pictures)
    
    for picture in pictures:
        filename = picture.split("/")[-1]
        results = model(picture)  
        for result in results:
            boxes = result.boxes
            masks = result.masks
            keypoints = result.keypoints
            probs = result.probs
            obb = result.obb
            result.show()
            result.save(filename=f"./datasets/results/{filename}")
        
