import argparse, os, glob, sys, json, random, tqdm
import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm
import math

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

sys.path.append('../vision/references/detection'.replace('/', os.sep))
import engine, utils
import common


def horizontal_flip(img, masks, boxes, p):
    if random.random() < p:
        img = img[:,::-1,:]
        for idx in range(masks.shape[0]):
            masks[idx] = masks[idx,:,::-1]
        boxes[:, [0, 2]] = img.shape[1] - boxes[:, [2, 0]]

    return img, masks, boxes

def random_crop(img, masks, boxes, p):
    if random.random() < p:
        buf_x = random.randint(0, 480)
        buf_y = random.randint(0, 270)
        crop_masks = np.zeros((masks.shape[0], 810, 1440), dtype=np.uint8)
        for idx in range(masks.shape[0]):
            distance_x = abs((boxes[idx][0] + boxes[idx][2]) / 2 - (buf_x + 720))
            distance_y = abs((boxes[idx][1] + boxes[idx][3]) / 2 - (buf_y + 405))
            size_x = (boxes[idx][2] - boxes[idx][0]) / 2 + 720
            size_y = (boxes[idx][3] - boxes[idx][1]) / 2 + 405
            if distance_x < size_x and distance_y < size_y:
	        # img
                crop_img = img[buf_y:buf_y+810, buf_x:buf_x+1440, :]
                img = cv.resize(crop_img, (img.shape[1], img.shape[0]))
	        # masks
                crop_masks[idx] = masks[idx, buf_y:buf_y+810, buf_x:buf_x+1440]
                masks[idx] = cv.resize(crop_masks[idx], (img.shape[1], img.shape[0]))
                # boxes
                boxes[idx][0] = max(buf_x, boxes[idx][0])
                boxes[idx][1] = max(buf_y, boxes[idx][1])
                boxes[idx][2] = min(buf_x+1440, boxes[idx][2])
                boxes[idx][3] = min(buf_y+810, boxes[idx][3])
    return img, masks, boxes

def illuminate(img, p):
    if random.random() < p:
        alpha = random.uniform(0.5, 1.5)
        beta = random.uniform(-50, 50)
        img = alpha * img + beta
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

class Dataset(object):
    def __init__(self, img_paths, annotation, is_train):
        self.img_paths = img_paths
        tmp = {}
        for k, v in annotation.items():
            tmp[v["filename"]] = v
        self.annotation = tmp
        self.is_train = is_train

    def __getitem__(self, idx):
        # load img
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path, 1)

        # load annotation
        regions = self.annotation[img_path.split(os.sep)[-1]]["regions"]
        num_objs = len(regions)

        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        masks = np.zeros((num_objs, img.shape[0], img.shape[1]), dtype=np.uint8)
        labels = np.zeros(num_objs, dtype=np.int64)

        for idx, region in enumerate(regions):
            tmp = region['shape_attributes']
            xs = tmp['all_points_x']
            ys = tmp['all_points_y']
            # bbox
            boxes[idx] = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
            # mask
            vertex = [[x, y] for x, y in zip(xs, ys)]
            cv2.fillPoly(masks[idx], [np.array(vertex)], 1)
            # label
            labels[idx] = list(region['region_attributes']['surface'].keys())[0]

        # data augmentation 21/12
        if self.is_train:
            img, masks, boxes = horizontal_flip(img, masks, boxes, p=0.5)
            img, masks, boxes = random_crop(img, masks, boxes, p=0.5)
            img = illuminate(img, p=0.5)
            if random.random() < 0.5:
                img = cv2.GaussianBlur(img, (5, 5), 0)

        img = img / 255.
        img = img.transpose(2,0,1)
        img = torch.from_numpy(img.astype(np.float32))
        boxes = torch.from_numpy(boxes)

        target = {
            "boxes": boxes,
            "masks": torch.from_numpy(masks),
            "labels": torch.from_numpy(labels),
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3]-boxes[:, 1]) * (boxes[:, 2]-boxes[:, 0]),
            "iscrowd": torch.zeros((num_objs,), dtype=torch.int64)
        }

        return img, target

def trainer(trainloader, model, optimizer):
    print("---------- Start Training ----------")
    
    try:
        with tqdm(trainloader, ncols=100) as pbar:
            train_loss = 0.0
            for images, targets in pbar:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)

                losses = sum(loss for loss in loss_dict.values())

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                loss_value = losses_reduced.item()

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    print(loss_dict_reduced)
                    sys.exit(1)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                train_loss += loss_value
        return train_loss
    except ValueError:
        pass

def validater(validloader, model):
    print("---------- Start Validating ----------")
    
    try:
        with tqdm(validloader, ncols=100) as pbar:
            valid_loss = 0.0
            for images, targets in pbar:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)

                losses = sum(loss for loss in loss_dict.values())

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                loss_value = losses_reduced.item()

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    print(loss_dict_reduced)
                    sys.exit(1)

                valid_loss += loss_value

        return valid_loss
    except ValueError:
        pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--evaluation', action='store_true')
    args = parser.parse_args()

    # open via json file
    tmp = open(common.ANNOTATION_FILE, 'r')
    annotation = json.load(tmp)
    tmp.close()

    # model and device
    model = common.get_model_instance_segmentation(common.NUM_CLASSES)
    model, device = common.setup_device(model)

    # dataset
    img_paths = glob.glob(common.TRAIN_ORG_IMGS)

    dataset = Dataset(img_paths, annotation, is_train=True)
    train_size = int(len(dataset) * 0.7)
    valid_size = len(dataset) - train_size
    train, valid = torch.utils.data.random_split(dataset, [train_size, valid_size])
    trainloader = torch.utils.data.DataLoader(train, batch_size=common.BATCH_SIZE,
            shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    validloader = torch.utils.data.DataLoader(valid, batch_size=common.BATCH_SIZE,
            shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=common.LEARNING_RATE)

    # tensorboard
    writer = SummaryWriter(log_dir="./logs")

    # Training
    early_stopping = [np.inf, 50, 0]
    for epoch in range(common.NUM_EPOCHS):
        # train
        train_loss = trainer(trainloader, model, optimizer)
        writer.add_scalar("Train Loss", train_loss, epoch + 1)
        # validate
        with torch.no_grad():
            valid_loss = validater(validloader, model)
        writer.add_scalar("Valid Loss", valid_loss, epoch + 1)

        # early stopping
        if valid_loss < early_stopping[0]:
            early_stopping[0] = valid_loss
            early_stopping[-1] = 0
            torch.save(model.state_dict(), common.SAVE_MODEL_DIR + str(epoch + 1))
            print(early_stopping)
        else:
            early_stopping[-1] += 1
            if early_stopping[-1] == early_stopping[1]:
                break
