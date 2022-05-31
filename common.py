
import torch
import torch.nn as nn
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

ITEM = 'tool', #'surface' 'hand'
# CLASS_NAMES = ['background', 'hand']
# CLASS_NAMES = ['background', 'main', 'assistant']
# CLASS_NAMES = ['background', 'forceps', 'tweezers', 'electrical-scalpel',
#                'scalpel', 'hook', 'syringe', 'needle-holder', 'pen']
CLASS_NAMES = ['background', 'muscle', 'adipose', 'dermal']
NUM_CLASSES = len(CLASS_NAMES)
NUM_EPOCHS = 1000
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
SAVE_MODEL_DIR ="./models/"
# ANNOTATION_FILE = "../hand-data/via_annotation_hand.json"
# ANNOTATION_FILE = "../tool_data/annotation_data_multi_tools2.json"
ANNOTATION_FILE = "../cutting_area_data/via_annotation_breast_surgery_parts.json"

# TRAIN_ORG_IMGS = "../hand-data/imgs/*.jpg"
# TRAIN_ORG_IMGS = "../tool_data/breast_surgery2/*.png"
TRAIN_ORG_IMGS = "../cutting_area_data/breast_surgery_annotation_images/*.png"

# TEST_IMG_PATH = "../main20200214_2/org_imgs2/*.png"
# TEST_IMG_PATH = "../hand-data/imgs/*.jpg"
TEST_IMG_PATH = "E:\/2022-04-05assistant/1/org_imgs/*.png"

# SAVE_COLOR_DIR = "../main20200214_2/area_color_mask/"
# SAVE_COLOR_DIR = "../main20200214_2/tool_color_mask/"
# SAVE_COLOR_DIR = "../main20200214_2/hand_color_mask/"
# SAVE_COLOR_DIR = "../hand-data/inf/"
SAVE_COLOR_DIR = "E:\/2022-04-05assistant/1/cutting_area_color_mask/"
# SAVE_BINARY_DIR = "../main20200214_2/tool_binary_mask/"

IMG_W = 960
IMG_H = 540

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
