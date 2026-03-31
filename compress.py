import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.compress import DetectionCompressor, DetectionFinetune


def compress(param_dict):
    with open(param_dict['sl_hyp'], errors='ignore') as f:
        sl_hyp = yaml.safe_load(f)
    param_dict.update(sl_hyp)
    param_dict['name'] = f'{param_dict["name"]}-prune'
    param_dict['patience'] = 0
    compressor = DetectionCompressor(overrides=param_dict)
    # compressor = SegmentationCompressor(overrides=param_dict)
    # compressor = PoseCompressor(overrides=param_dict)
    # compressor = OBBCompressor(overrides=param_dict)
    prune_model_path = compressor.compress()
    return prune_model_path

def finetune(param_dict, prune_model_path):
    param_dict['model'] = prune_model_path
    param_dict['name'] = f'{param_dict["name"]}-finetune'
    trainer = DetectionFinetune(overrides=param_dict)
    # trainer = SegmentationFinetune(overrides=param_dict)
    # trainer = PoseFinetune(overrides=param_dict)
    # trainer = OBBFinetune(overrides=param_dict)
    trainer.train()


if __name__ == '__main__':
    param_dict = {
        # origin

        'model': r'E:\tunnel_crack\SCI_tunnel_crack\loss\TCD-YOLO-best.pt',
        'data':'cfg/datasets/tunnel_crack.yaml',
        'imgsz': 512,
        'epochs': 100,
        'batch': 16,
        'workers': 8,
        'cache': False,
        'optimizer': 'SGD',
        'device': '1',
        'hsv_h':0.025, 
        'hsv_s':0.7,  
        'hsv_v':0.6,  

        'project':'runs/prune',
        'name':'yolov11_sgd_speedup1.6_nws_lamp_road_crack_test',
        
#         'prune_method':'l1',
#         'prune_method':'group_norm',    
#         'prune_method':'lamp'

        'prune_method':'lamp',
        'global_pruning': True,
        'speed_up': 1.6,
        'reg': 0.0005,  
        'sl_epochs': 400,  
        'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
        'sl_model': None,
    }
    
    prune_model_path = compress(copy.deepcopy(param_dict))
    finetune(copy.deepcopy(param_dict), prune_model_path)
