import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image

import os, sys
sys.path.append('/home/yzhang94/anaconda3/envs/spectre8/lib/python3.8/site-packages')


import argparse
import torch
import numpy as np
from skimage.transform import estimate_transform, warp
from tqdm import tqdm
sys.path.append('/home/yzhang94/Documents/ros2_ws/src/python_service/python_service')
from datasets.data_utils import landmarks_interpolate

# sys.path.append('/home/yzhang94/Documents/ros2_ws/src/python_service/python_service/external/face_detection/ibug/face_detection/retina_face')
# from retina_face import RetinaFace
# import torchvision.models as models


from external.Visual_Speech_Recognition_for_Multiple_Languages.tracker.face_tracker import FaceTracker
from external.Visual_Speech_Recognition_for_Multiple_Languages.tracker.utils import get_landmarks
from src.spectre import SPECTRE
from config import cfg as spectre_cfg
from src.utils.util import tensor2video
import torchvision
import fractions
import librosa
from moviepy.editor import AudioFileClip
from scipy.io import wavfile
import collections
import gc

print(torch.cuda.is_available())

class VideoSubscriber(Node):
    def __init__(self):
        super().__init__('video_frame_subscriber')
        
        # Initialize subscriber to the 'video_frames' topic
        self.subscription = self.create_subscription(
            CompressedImage,
            'video_frames',
            self.frame_callback,
            10)
        
        image_path = '/home/yzhang94/Documents/ros2_ws/src/python_service/python_service/samples/LRS3/PbgB2TaYhio_00007/000000.jpg'
        image = cv2.imread(image_path)

        # print(image)
        # cv2.imshow("Received Video Stream",image)

        # Wait for a short time to allow OpenCV to display the image
        cv2.waitKey(1)
        self.crop_face = True
        self.input = '/home/yzhang94/Documents/ros2_ws/src/python_service/python_service/samples/LRS3/PbgB2TaYhio_00007.mp4'
        folderpath = '/home/yzhang94/Documents/ros2_ws/src/python_service/python_service/samples/LRS3/PbgB2TaYhio_00007'
        spectre_cfg.pretrained_modelpath = "/home/yzhang94/Documents/ros2_ws/src/python_service/python_service/pretrained/spectre_model.tar"
        spectre_cfg.model.use_tex = False
        self.device = 'cuda'
        self.spectre = SPECTRE(spectre_cfg, 'cuda')
        self.spectre.eval()
        self.face_tracker = FaceTracker()
        self.fps = 25

        # Create a CvBridge object to convert ROS images to OpenCV format
        self.bridge = CvBridge()
        self.image_publisher = self.create_publisher(Image, 'display_image', 10)
        # image_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        # self.image_publisher.publish(image_msg)
        
        # Frame counter
        self.frame_count = 0

        self.get_logger().info("Video frame subscriber node has been started.")



    def frame_callback(self, msg):
        # Increment the frame counter
        self.frame_count += 1
        self.get_logger().info(f"Received frame number: {self.frame_count}")

        # Convert the ROS CompressedImage message to an OpenCV image
        frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        image_path = '/home/yzhang94/Documents/ros2_ws/src/python_service/python_service/samples/LRS3/PbgB2TaYhio_00007/000000.jpg'
        # image = cv2.imread(image_path)
        cv2.imwrite(image_path, frame)

        imagepath_list = []
        face_info = collections.defaultdict(list)
        detected_faces = self.face_tracker.face_detector(frame, rgb=False)
        landmarks, scores = self.face_tracker.landmark_detector(frame, detected_faces, rgb=False)
        face_info['bbox'].append(detected_faces)
        face_info['landmarks'].append(landmarks)
        face_info['landmarks_scores'].append(scores)
        imagepath_list.append(image_path)
        landmarks = get_landmarks(face_info)

        
        print(f'the frame rate is {self.fps}')
        if self.crop_face:
            landmarks = landmarks_interpolate(landmarks)
            if landmarks is None:
                print('No faces detected in input {}'.format(self.input))
                return

        original_video_length = len(imagepath_list)
        imagepath_list.insert(0, imagepath_list[0])
        imagepath_list.insert(0, imagepath_list[0])
        imagepath_list.append(imagepath_list[-1])
        imagepath_list.append(imagepath_list[-1])

        landmarks.insert(0, landmarks[0])
        landmarks.insert(0, landmarks[0])
        landmarks.append(landmarks[-1])
        landmarks.append(landmarks[-1])

        image_paths = imagepath_list
        landmarks = np.array(landmarks)
        L = 50
        indices = list(range(len(image_paths)))
        overlapping_indices = [indices[i: i + L] for i in range(0, len(indices), L - 4)]

        if len(overlapping_indices[-1]) < 5:
            overlapping_indices[-2] = overlapping_indices[-2] + overlapping_indices[-1]
            overlapping_indices[-2] = np.unique(overlapping_indices[-2]).tolist()
            overlapping_indices = overlapping_indices[:-1]

        overlapping_indices = np.array(overlapping_indices)
        image_paths = np.array(image_paths)
        all_shape_images = []
        all_images = []

        with torch.no_grad():
            for chunk_id in range(len(overlapping_indices)):
                # print(f'Processing frames {overlapping_indices[chunk_id][0]} to {overlapping_indices[chunk_id][-1]}')
                image_paths_chunk = image_paths[overlapping_indices[chunk_id]]
                landmarks_chunk = landmarks[overlapping_indices[chunk_id]] if self.crop_face else None

                images_list = []

                for j in range(len(image_paths_chunk)):
                    frame = cv2.imread(image_paths_chunk[j])
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    kpt = landmarks_chunk[j]

                    tform = self.ccrop_face(frame, kpt, scale=1.6)
                    cropped_image = warp(frame, tform.inverse, output_shape=(224, 224))
                    images_list.append(cropped_image.transpose(2, 0, 1))

                images_array = torch.from_numpy(np.array(images_list)).float().to(self.device)

                # Free up memory here to avoid GPU memory overflow
                torch.cuda.empty_cache()
                gc.collect()

                codedict, initial_deca_exp, initial_deca_jaw = self.spectre.encode(images_array)
                codedict['exp'] += initial_deca_exp
                codedict['pose'][..., 3:] += initial_deca_jaw

                for key in codedict.keys():
                    if chunk_id == 0 and chunk_id == len(overlapping_indices) - 1:
                        pass
                    elif chunk_id == 0:
                        codedict[key] = codedict[key][:-2]
                    elif chunk_id == len(overlapping_indices) - 1:
                        codedict[key] = codedict[key][2:]
                    else:
                        codedict[key] = codedict[key][2:-2]

                opdict, visdict = self.spectre.decode(codedict, rendering=True, vis_lmk=False, return_vis=True)
                all_shape_images.append(visdict['shape_images'].detach().cpu())
                all_images.append(codedict['images'].detach().cpu())

                # Clear CUDA cache and perform garbage collection after each chunk
                torch.cuda.empty_cache()
                gc.collect()
            # print(11111)

        vid_shape = tensor2video(torch.cat(all_shape_images, dim=0))[2:-2]
        vid_orig = tensor2video(torch.cat(all_images, dim=0))[2:-2]
        grid_vid = np.concatenate((vid_shape, vid_orig), axis=2)


        # Display the frame with the frame count
        # cv2.imshow("Received Video Stream", all_images[0].numpy().astype(np.uint8))
        image_msg = self.bridge.cv2_to_imgmsg(grid_vid[0].astype(np.uint8), encoding="bgr8")
        self.image_publisher.publish(image_msg)
        # cv2.imshow("Received Video Stream",image)
        
        # Wait for a short time to allow OpenCV to display the image
        # cv2.waitKey(1)
    
    def ccrop_face(self, frame, landmarks, scale=1.0):
        image_size = 224
        left, right = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
        top, bottom = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])

        h, w, _ = frame.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size * scale)

        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2],
                            [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        return tform

def main(args=None):

    rclpy.init(args=args)
    video_subscriber = VideoSubscriber()
    
    try:
        rclpy.spin(video_subscriber)
    except KeyboardInterrupt:
        video_subscriber.get_logger().info("Video frame subscriber node has been stopped.")
    finally:
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        video_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
