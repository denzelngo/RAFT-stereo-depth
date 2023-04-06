from cgi import print_environ
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Object points in 3D
GRID_SHAPE = (9, 6)
objp = np.zeros((GRID_SHAPE[0] * GRID_SHAPE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:GRID_SHAPE[0], 0:GRID_SHAPE[1]].T.reshape(-1, 2)
objp *= 0.026  # One square on my grid has 26mm
# print(objp)
SAMPLE_RATE = 2
print('Processing Left camera ...')
FOLDER = "/home/user5/Downloads/ov9281_video12/left/"
fnames = os.listdir(FOLDER)[::SAMPLE_RATE]
obj_pts = []
img_pts = []

TMP = "tmp/left/"

imgsz = (640, 400)

for fname in fnames:
    print(f"processing {fname}")
    img = Image.open(FOLDER + fname).resize(imgsz)
    arr = np.array(img)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ret, corners = cv2.findChessboardCorners(arr, GRID_SHAPE, None)

    arr_vis = cv2.drawChessboardCorners(arr, GRID_SHAPE, corners, ret)

    if ret:
        cv2.imwrite(TMP + fname, arr_vis)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_subpix = cv2.cornerSubPix(arr, corners, (11, 11), (-1, -1), criteria)
        obj_pts.append(objp)
        img_pts.append(corners_subpix)
print('Get calibration parameters ...')
ret, K_l, dist_coeff_l, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (arr.shape[1], arr.shape[0]), None, None)
print('Intrinsic param matrix: ')
print(K_l)
print('lens distortion coefficients')
print(dist_coeff_l)
# print('rvecs')
# print(rvecs)
# print('tvecs')
# print(tvecs)

print('Processing Right camera ...')
FOLDER = "/home/user5/Downloads/ov9281_video12/right/"
fnames = os.listdir(FOLDER)[::SAMPLE_RATE]
obj_pts = []
img_pts = []

TMP = "tmp/right/"

for fname in fnames:
    print(f"processing {fname}")
    img = Image.open(FOLDER + fname).resize(imgsz)
    arr = np.array(img)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ret, corners = cv2.findChessboardCorners(arr, (9, 6), None)

    arr_vis = cv2.drawChessboardCorners(arr, GRID_SHAPE, corners, ret)

    if ret:
        cv2.imwrite(TMP + fname, arr_vis)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_subpix = cv2.cornerSubPix(arr, corners, (11, 11), (-1, -1), criteria)
        obj_pts.append(objp)
        img_pts.append(corners_subpix)
print('Get calibration parameters ...')
ret, K_r, dist_coeff_r, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (arr.shape[1], arr.shape[0]), None, None)

print('Intrinsic param matrix: ')
print(K_r)
print('lens distortion coefficients')
print(dist_coeff_r)

np.save("K_l.npy", K_l)
np.save("K_r.npy", K_r)

np.save("dist_coeff_l.npy", dist_coeff_l)
np.save("dist_coeff_r.npy", dist_coeff_r)
