import cv2
import glob
import os
import numpy as np
from tqdm import tqdm

import common

org_paths = glob.glob(common.TEST_IMG_PATH)
org_paths.sort()
mask_paths = glob.glob(common.SAVE_COLOR_DIR + "*.png")
mask_paths.sort()

#gaze_points = np.loadtxt("./gaze.csv", delimiter=",", skiprows=1, usecols=(1, 2))
#gaze_points *= np.array([960, 540])

with tqdm(org_paths, ncols=100) as pbar:
	i = 0
	for org_path in pbar:
		org_img = cv2.imread(org_path)
		org_img = cv2.resize(org_img, (960, 540))
		mask_img = cv2.imread(mask_paths[i])
		mask_img = cv2.resize(mask_img, (960, 540))
		blend_img = cv2.addWeighted(org_img, 1, mask_img, 0.5, 0)
		i += 1

#	gaze_point = (int(gaze_points[i][0]), int(gaze_points[i][1]))
#	cv2.drawMarker(org_img, gaze_point, (0, 0, 255),
#		       markerType=cv2.MARKER_CROSS, markerSize=50,
#		       thickness=10, line_type=cv2.LINE_8)

		cv2.imwrite("../main20200214_1/tool_hand_blend/" + os.path.basename(org_path), blend_img)
# cv2.imwrite("E:/2022-06-28mainMsIshikawa/gaze_blend/" + os.path.basename(org_path), blend_img)
