import time
import cv2
import numpy as np
import tensorrt as trt
from threading import Thread
import os
import pycuda.driver as cuda
import pycuda.autoinit
# from camera_opencv import Camera
from camera_ov9281_opencv import Camera
#from camera_gst import Camera
from PIL import Image
trt.init_libnvinfer_plugins(None, '')


H_rect = 400
W_rect = 640

SHAPE_IN = (320, 200)
SHAPE_OUT = (320, 200)

class ExposureAjust(Thread):
    def __init__(self):
        super().__init__()
        self.expo = 681
        self.running = True
    def run(self):
        while self.running:
            cmd = f'v4l2-ctl -c exposure={self.expo}'
            # subprocess.call(cmd, shell=True)
            os.system(cmd)
            # print(cmd)
            # time.sleep(3)

    def stop(self):
        self.running = False

def get_calibration() -> tuple:
    fs = cv2.FileStorage(
        "rectify_ov9281_neg_alpha.yaml", cv2.FILE_STORAGE_READ
    )

    map_l = (fs.getNode("map_l_1").mat(), fs.getNode("map_l_2").mat())

    map_r = (fs.getNode("map_r_1").mat(), fs.getNode("map_r_2").mat())

    fs.release()

    return map_l, map_r

def resize(frame, dst_width):
    width = frame.shape[1]
    height = frame.shape[0]
    scale = dst_width * 1.0 / width
    return cv2.resize(frame, (int(scale * width), int(scale * height)))

def allocate_buffers(engine):
    host_inputs = []
    cuda_inputs = []
    host_outputs = []
    cuda_outputs = []
    for binding in engine:
        print(engine.get_binding_shape(binding))
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        print(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        # bindings.append(int(cuda_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            host_inputs.append(host_mem)
            cuda_inputs.append(cuda_mem)
        else:
            host_outputs.append(host_mem)
            cuda_outputs.append(cuda_mem)
    stream = cuda.Stream()
    return host_inputs, cuda_inputs, host_outputs, cuda_outputs, stream


if __name__ == "__main__":

    cam = Camera(0)
    # mapl, mapr = get_calibration()

    expo_adjust = False
    if expo_adjust:
        expo_adjust = ExposureAjust()
        expo_adjust.start()
        new_expo= expo_adjust.expo

    # Crop parameters
    dx = 25
    dy = int(dx/W_rect*H_rect)


    cuda.init()
    dev = cuda.Device(0)
    ctx = dev.make_context()

    show_original_image = True
    save_frame = False

    check_time = 0


    if save_frame:
        count = 0

    if show_original_image:
        cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera', 1000, 500)


    trtLogger = trt.Logger(trt.Logger.INFO)
    trt_engine_path = 'raft_faster_gray.engine'
    with open(trt_engine_path, 'rb') as f, trt.Runtime(trtLogger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

        h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)

    


        # cam_l = CameraThread(0, img_size)
        # cam_r = CameraThread(1, img_size)

        with engine.create_execution_context() as context:
            try:
                while True:

                    # print("Capturing. Press spacebar to capture an image pair.%%")



                    tic = time.time()
                    img_l, img_r = cam.read()

                    # frame = cam.read()
                    # frame = resize(frame, 1280.0)
                    # frame = cv2.flip(frame, 0)
                    # frame = cv2.flip(frame, 1)
                    # img_l = frame[:, :640].copy()
                    # img_r = frame[:, 640:].copy()

                    # Exposure adjustment 
                    if expo_adjust:
                        if check_time == 60:
                            whiteness =  np.mean(img_l)
                            d_exposure = (whiteness-100)/500*new_expo
                            if d_exposure >= 0:
                                d_exposure = int(min(100, d_exposure))
                                d_exposure = int(max(1, d_exposure))
                            else:
                                d_exposure = int(max(-100, d_exposure))
                                d_exposure = int(min(-1, d_exposure))
                            new_expo -= d_exposure
                            new_expo = max(5, new_expo)
                            expo_adjust.expo = new_expo
                            check_time = 0
                        check_time +=1



                    # tic_rect = time.time()

                    # img_l = cv2.remap(img_l, *mapl, cv2.INTER_LINEAR)
                    # img_r = cv2.remap(img_r, *mapr, cv2.INTER_LINEAR)
                    # fps_rect = 1/(time.time() - tic_rect)

                    img_l = cv2.cvtColor(img_l, cv2.COLOR_GRAY2RGB)
                    img_r = cv2.cvtColor(img_r, cv2.COLOR_GRAY2RGB)

                    img_l = img_l[dy:H_rect-dy, dx: W_rect-dx, :]
                    img_r = img_r[dy:H_rect-dy, dx: W_rect-dx, :]


                    img_l = cv2.resize(img_l, SHAPE_IN)
                    img_r = cv2.resize(img_r, SHAPE_IN)

                    # img_l = Image.fromarray(img_l).resize(img_size)
                    # img_r = Image.fromarray(img_r).resize(img_size)

                    # img_l = Image.fromarray(img_l)
                    # img_r = Image.fromarray(img_r)

                    # r = int(img_l.shape[0] / img_size[1])
                    # img_l = img_l.reshape((img_size[1], r, img_size[0], r, 3)).max(3).max(1)
                    # img_r = img_r.reshape((img_size[1], r, img_size[0], r, 3)).max(3).max(1)

                    if save_frame:
                        count += 1
                        cv2.imwrite(f'tmp/left/{str(count).zfill(4)}.jpg', img_l)
                        cv2.imwrite(f'tmp/right/{str(count).zfill(4)}.jpg', img_r)


                    if show_original_image:

                        img_lr = np.hstack([img_l.copy(), img_r.copy()])
                        # img_lr = cv2.resize(img_lr, (1920, 1080//2))
                    
                    
                    # img_l = np.array(img_l)[:,:,::-1].astype(np.float16)
                    # img_r = np.array(img_r)[:,:,::-1].astype(np.float16)

                    img_l = np.array(img_l).astype(np.float16)
                    img_r = np.array(img_r).astype(np.float16)

                    if not 'raft' in trt_engine_path:
                        img_l /= 255.
                        img_r /= 255.

                    img_l = np.transpose(img_l, [2,0,1])
                    img_r = np.transpose(img_r, [2,0,1])

                    img_l = np.expand_dims(img_l, axis=0)
                    img_r = np.expand_dims(img_r, axis=0)

                    img_l = np.ascontiguousarray(img_l)
                    img_r = np.ascontiguousarray(img_r)

                    np.copyto(h_input[0], img_l.ravel())
                    np.copyto(h_input[1], img_r.ravel())

                    # Transfer input data to the GPU.
                    cuda.memcpy_htod_async(d_input[0], h_input[0], stream)
                    cuda.memcpy_htod_async(d_input[1], h_input[1], stream)
                    # Run inference.
                    context.execute_async_v2(bindings=[int(d_input[0]), int(d_input[1]) ,int(d_output[0])], stream_handle=stream.handle)
                    # Transfer predictions back from the GPU.
                    cuda.memcpy_dtoh_async(h_output[0], d_output[0], stream)
                    # Synchronize the stream
                    stream.synchronize()
                    # Post-processing
                    disp = h_output[0]
                    disp = disp.reshape(SHAPE_OUT[1], SHAPE_OUT[0])
                    # disp = disp.reshape(184,320)


                    toc = time.time()

                    
                    if 'raft' in trt_engine_path:
                        pred_disp = disp.copy() * (-1.0) 
                    dist = 2530*8/(pred_disp.max()*(1920/320))
                    print(f'Nearest point: {dist:.4f} cm; {1/(toc - tic):.4f} FPS')
                    
                    pred_disp = (pred_disp - pred_disp.min())/(pred_disp.max()-pred_disp.min())
                                  
                    depth = cv2.applyColorMap((pred_disp*255.0).astype(np.uint8), cv2.COLORMAP_JET)
                    # pred_disp = disp.clip(0, 192)
                    # depth = cv2.applyColorMap((pred_disp*255.0/192).astype(np.uint8), cv2.COLORMAP_JET)
                    depth = cv2.resize(depth, (640, 400))

                    # img_lr = np.hstack([img_l, img_r])
                    # img_lr = cv2.resize(img_lr, (1920, 1080//2))
                    
                    cv2.imshow("Depth", depth)
                    if show_original_image:
                        cv2.imshow("Camera", img_lr)

                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        break
            except KeyboardInterrupt as e:
                print("closing")
                cam.release()
                if expo_adjust:
                    expo_adjust.stop()

                ctx.pop()
                del ctx
            finally:
                # cam_l.stop()
                # cam_r.stop()
                cam.release()
                if expo_adjust:
                    expo_adjust.stop()

                ctx.pop()
                del ctx
