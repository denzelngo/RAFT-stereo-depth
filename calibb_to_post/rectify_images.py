import cv2
import os


def get_calibration() -> tuple:
    fs = cv2.FileStorage(
        "rectify_ov9281_neg_alpha.yaml", cv2.FILE_STORAGE_READ
    )

    map_l = (fs.getNode("map_l_1").mat(), fs.getNode("map_l_2").mat())

    map_r = (fs.getNode("map_r_1").mat(), fs.getNode("map_r_2").mat())

    fs.release()

    return map_l, map_r


# folder = '/home/user5/Downloads/video3/left'
# folder = '/home/user5/Downloads/ov9281_slam_video6/left'
folder = '/home/user5/Downloads/slam_session_16Fev23/ov9281_slam_video1/left'
folder_rect = 'slam_session_16Fev23/ov9281_slam_rect1/'

if not os.path.isdir(folder_rect):
    os.mkdir(folder_rect)
    os.mkdir(folder_rect + 'left/')
    os.mkdir(folder_rect + 'right/')

mapl, mapr = get_calibration()
imgs = os.listdir(folder)

for img in imgs:
    print(img)
    path_l = os.path.join(folder, img)
    path_r = path_l.replace('left', 'right')

    imgl = cv2.imread(path_l)
    imgr = cv2.imread(path_r)

    # imgl = cv2.resize(imgl, (960, 540))
    # imgr = cv2.resize(imgr, (960, 540))

    imgl_rect = cv2.remap(imgl, *mapl, cv2.INTER_LINEAR)
    imgr_rect = cv2.remap(imgr, *mapr, cv2.INTER_LINEAR)
    # imgl_rect = cv2.remap(imgl, *mapl, cv2.INTER_LANCZOS4)
    # imgr_rect = cv2.remap(imgr, *mapr, cv2.INTER_LANCZOS4)

    path_rect_l = os.path.join(folder_rect, 'left', img)
    path_rect_r = path_rect_l.replace('left', 'right')

    cv2.imwrite(path_rect_l, imgl_rect)
    cv2.imwrite(path_rect_r, imgr_rect)
