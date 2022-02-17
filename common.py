
import torch
import torch.nn as nn
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# CLASS_NAMES = ['background', 'tool']
# CLASS_NAMES = ['background', 'main', 'assistant']
CLASS_NAMES = ['background', 'forceps', 'tweezers', 'electrical-scalpel',
               'scalpels', 'hook', 'syringe', 'needle-holder', 'pen']
# CLASS_NAMES = ['background', 'muscle', 'adipose', 'dermal']
NUM_CLASSES = len(CLASS_NAMES)
NUM_EPOCHS = 1000
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
SAVE_MODEL_DIR ="./models/"
# ANNOTATION_FILE = "../cutting_area_data/via_annotation_breast_surgery_parts.json"
ANNOTATION_FILE = "../tool_data/via_annotation_tools.json"

# TRAIN_ORG_IMGS = "../cutting_area_data/breast_surgery/*.png"
TRAIN_ORG_IMGS = "../tool_data/org_imgs/*.png"
TEST_IMG_PATH = "../main20200214_2/org_imgs/*.png"
SAVE_COLOR_DIR = "../main20200214_2/tool_color_mask/"
SAVE_BINARY_DIR = "../main20200214_2/tool_binary_mask/"

IMG_W = 960
IMG_H = 540

TRAIN_DATA_IMGS = "../data/tool/annotation_imgs/*.png"
TEST_DATA_IMGS = "../main20170707/org_imgs/*.png"
SAVE_NPY_DIR = "../main20170707/multi_channel_tool/"
SAVE_DIR = "../data/tool/masked/"


def setup_device(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        print("---------- Use ", torch.cuda.device_count(), "GPUs ----------")
        model = nn.DataParallel(model)
    else:
        print("---------- Use CPU ----------")
    model.to(device)

    return model, device

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    
    return model
