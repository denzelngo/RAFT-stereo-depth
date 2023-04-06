import os
import cv2
import numpy as np

from PIL import Image

SAMPLE_RATE = 2
imgsz = (640, 400)

# Object points in 3D
GRID_SHAPE = (9, 6)
objp = np.zeros((GRID_SHAPE[0] * GRID_SHAPE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:GRID_SHAPE[0], 0:GRID_SHAPE[1]].T.reshape(-1, 2)
objp *= 26  # One square on my grid has 26mm

folder_right = "/home/user5/Downloads/ov9281_video12/right/"
folder_left = "/home/user5/Downloads/ov9281_video12/left/"

(K_l, K_r, dist_l, dist_r) = np.load("K_l.npy"), np.load("K_r.npy"), np.load("dist_coeff_l.npy"), np.load(
    "dist_coeff_r.npy")

obj_pts = []
img_pts_l, img_pts_r = [], []

for fname_l, fname_r in zip(sorted(os.listdir(folder_left)[::SAMPLE_RATE]),
                            sorted(os.listdir(folder_right)[::SAMPLE_RATE])):

    print(f"processing {fname_l, fname_r}")
    img_l, img_r = Image.open(folder_left + fname_l).resize(imgsz), \
                   Image.open(folder_right + fname_r).resize(imgsz)
    arr_l, arr_r = np.array(img_l), np.array(img_r)
    gray_l, gray_r = arr_l, arr_r  # the images are already grayscale so no conversion

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, GRID_SHAPE, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, GRID_SHAPE, None)

    if ret_l and ret_r:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_subpix_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        corners_subpix_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

        obj_pts.append(objp)
        img_pts_l.append(corners_subpix_l)
        img_pts_r.append(corners_subpix_r)

flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
print('Stereo calibration... ')
retS, K_l, dist_l, K_r, dist_r, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(
    obj_pts,
    img_pts_l,
    img_pts_r,
    K_l, dist_l,
    K_r, dist_r,
    gray_l.shape[::-1],
    criteria_stereo,
    flags)

print('Kl: ', K_l)
print('dist l: ', dist_l)
print('Kr: ', K_r)
print('dist r: ', dist_r)
print('R: ', Rot)
print('T: ', Trns)
# print('Stereo rectification... ')
# # rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(K_l, dist_l, K_r, dist_r, gray_l.shape[::-1], Rot, Trns, 1 ,(0,0))
# rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(K_l, dist_l, K_r, dist_r, gray_l.shape[::-1],
#                                                                           Rot, Trns, flags=cv2.CALIB_ZERO_DISPARITY,
#                                                                           alpha=-1)
#
# # left_stereo_maps = cv2.initUndistortRectifyMap(K_l, dist_l, rect_l, proj_mat_l,
# #                                                gray_l.shape[::-1], cv2.CV_16SC2)
# # right_stereo_maps = cv2.initUndistortRectifyMap(K_r, dist_r, rect_r, proj_mat_r,
# #                                                 gray_l.shape[::-1], cv2.CV_16SC2)
#
# left_stereo_maps = cv2.initUndistortRectifyMap(K_l, dist_l, rect_l, proj_mat_l,
#                                              gray_l.shape[::-1], cv2.CV_32FC1)
# right_stereo_maps = cv2.initUndistortRectifyMap(K_r, dist_r, rect_r, proj_mat_r,
#                                               gray_l.shape[::-1], cv2.CV_32FC1)
#
# cv_file = cv2.FileStorage("rectify_ov9281_neg_alpha_320x200.yaml", cv2.FILE_STORAGE_WRITE)
# cv_file.write("map_l_1", left_stereo_maps[0])
# cv_file.write("map_l_2", left_stereo_maps[1])
# cv_file.write("map_r_1", right_stereo_maps[0])
# cv_file.write("map_r_2", right_stereo_maps[1])
# cv_file.release()
