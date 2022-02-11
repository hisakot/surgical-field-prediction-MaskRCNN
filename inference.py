import argparse
import cv2
import glob
import numpy as np
import os

import torch
import torchvision
from tqdm import tqdm

import common

def get_coloured_mask(mask, pred_cls, boxes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
# colours = [[0, 0, 0],[0, 255, 0],[0, 255, 255],[255, 255, 0],[80, 70, 180],[180, 40, 250],[245, 145, 50],[70, 150, 250],[50, 190, 190]] # black, green, yellow, cyan
    colours = [[0, 0, 0],[0, 255, 0],[255, 0, 0],[0, 255, 255]] # black, green, blue, cyan
    b[mask == 1], g[mask == 1], r[mask == 1] = colours[common.CLASS_NAMES.index(pred_cls)]
    coloured_mask = np.stack([b, g, r], axis=2)
    
    b, g, r = colours[CLASS_NAMES.index(pred_cls)]
    cv2.rectangle(img, (boxes[0]), (boxes[1]), (b, g, r), thickness=2)
    return coloured_mask

def get_binary_mask(mask, pred_cls):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    binary = [255, 0, 0] # black, green, blue, cyan
    b[mask == 1], g[mask == 1], r[mask == 1] = binary
    binary_mask = np.stack([b, g, r], axis=2)

    return binary_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_name', required=True)
    args = parser.parse_args()

    # model and device
    best_model = common.SAVE_MODEL_DIR + args.model_name
    model = common.get_model_instance_segmentation(common.NUM_CLASSES)
    model, device = common.setup_device(model)
    model.load_state_dict(torch.load(best_model, map_location=device))
    model.eval()

    # data loader
    img_paths = glob.glob(common.TEST_IMG_PATH)
    print(range(len(img_paths)))

    # Prediction
    confidence = 0.5
    for idx in tqdm(range(len(img_paths))):
        img_path = img_paths[idx]
        img = cv2.imread(img_path)
        img = img / 255.
        img = img.transpose(2,0,1)
        img = torch.from_numpy(img.astype(np.float32))

        pred = model([img.to(device)])

        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > confidence]
        masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
        if masks.ndim == 2:
            masks = masks.reshape([1, masks.shape[0], masks.shape[1]])
        pred_class = [common.CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
        if len(pred_t) == 0:
            masks = []
            boxes = []
            pred_cls = []
        else:
            pred_t = pred_t[-1]
            masks = masks[:pred_t+1]
            boxes = pred_boxes[:pred_t+1]
            pred_cls = pred_class[:pred_t+1]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        binary_mask = np.zeros((img.shape[0], img.shape[1], common.NUM_CLASSES))
        color_mask = np.zeros((common.IMG_W, common.IMG_H))

        muscle = 0
        adipose = 0
        dermal = 0

        for i in range(len(masks)):
            if pred_cls[i] == "muscle":
                muscle += np.count_nonzero(masks[i] == True)
            elif pred_cls[i] == "adipose":
                adipose += np.count_nonzero(masks[i] == True)
            elif pred_cls[i] == "dermal":
                dermal += np.count_nonzero(masks[i] == True)

            rgb_mask = get_coloured_mask(masks[i], pred_cls[i], boxes[i])
            color_mask = cv2.addWeighted(rgb_mask, 1, rgb_mask, 1, 0)

            bi_mask = get_binary_mask(masks[i], pred_cls[i])
            binary_mask[:, :, common.CLASS_NAMES.index(pred_cls)] = bi_mask[0]

        # save predicted mask
        color_mask = cv2.resize(color_mask, (common.IMG_W, common.IMG_H))
        cv2.imwrite(common.SAVE_COLOR_DIR + img_paths[idx].split(os.sep)[-1], color_mask)
        binary_mask = cv2.resize(binary_mask, (common.IMG_W, common.IMG_H))
        cv2.imwrite(common.SAVE_BINARY_DIR + img_paths[idx].split(os.sep)[-1], binary_mask)

        # calculate area size
        whole = common.IMG_W * common.IMG_H
        area = np.array([[idx+1, muscle / whole * 100, adipose / whole * 100, dermal / whole * 100]])
        print(area)
        with open("opened_area.csv", "a") as f:
            np.savetxt(f, area, delimiter=",", fmt="%.4f")
