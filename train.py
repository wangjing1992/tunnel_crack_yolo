import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


def main():
    model = YOLO('E:\\tunnel_crack\\py\\ultralytics-yolo11-main\\ultralytics\\cfg\\models\\11\\yolo11-FDPN-TADDH.yaml')  # load a pretrained model (recommended for training)
    #model.info()
    model.train(data='cfg/datasets/tunnel_crack.yaml',
                cache=False,
                imgsz=512,
                lr0=0.001,
                lrf=0.01,
                augment=True,
                cos_lr=False,
                epochs=100,
                batch=16,
                iou =0.7,
                device=1,
                save_period=10,
                seed=42,  
                patience=50,
                optimizer='AdamW', 
                project='runs/train',
                weight_decay=0.0005,  
                name='yolov11n_FDPN-TADDH_ciou_nwd_exp',
                workers=8, 
                hsv_h=0.025, 
                hsv_s=0.7, 
                hsv_v=0.6,  
                )
    

if __name__ == '__main__':
    main()

