import warnings

from ibug.face_alignment import FANPredictor
from ibug.face_detection import RetinaFacePredictor

warnings.filterwarnings("ignore")
import math
import numpy as np


class LandmarksDetector:
    def __init__(self, device, batch_size, model_name="resnet50"):
        self.face_detector = RetinaFacePredictor(
            device=device,
            threshold=0.8,
            model=RetinaFacePredictor.get_model(model_name),
        )
        self.landmark_detector = FANPredictor(device=device, model=None)
        self.batch_size = batch_size

    def __call__(self, video_frames):
        landmarks = []
        batch = self.batch_size
        step = math.ceil(len(video_frames)/batch)

        for i in range(step):
            video_frame_local = video_frames[i*batch:(i+1)*batch]
            if len(video_frame_local) < 1:
                break

            detected_faces_list = self.face_detector(video_frame_local, rgb=False)
            if len(detected_faces_list) != len(video_frame_local):
                detected_faces_list = self.face_detector(video_frame_local, rgb=False, old=True)

            can_batch = True # debug
            for j, detected_faces in enumerate(detected_faces_list):
                if len(detected_faces) != 1:
                    can_batch = False
                    break
            
            if can_batch:
                detected_faces_list = np.array(detected_faces_list)
                face_points_list, _ = self.landmark_detector(video_frame_local, detected_faces_list, rgb=True)
                landmarks.extend(face_points_list)
            else:
                for j, detected_faces in enumerate(detected_faces_list):
                    if len(detected_faces) == 0:
                        landmarks.append(None)
                    else:
                        face_points, _ = self.landmark_detector(video_frame_local[j], detected_faces, rgb=True, old=True)
                        if face_points.shape[0] == 1:
                            landmarks.append(face_points[0])
                        else:
                            max_id, max_size = 0, 0
                            for idx, bbox in enumerate(detected_faces):
                                bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                                if bbox_size > max_size:
                                    max_id, max_size = idx, bbox_size
                            landmarks.append(face_points[max_id])

        return landmarks


class PureFeatureDetector:
    def __init__(self, device, batch_size, model_name="resnet50"):
        self.face_detector = RetinaFacePredictor(
            device=device,
            threshold=0.8,
            model=RetinaFacePredictor.get_model(model_name),
        )
        self.bs = batch_size

    def __call__(self, video_frames):
        landmarks = []
        batch = self.bs
        step = math.ceil(len(video_frames)/batch)
        for i in range(step):
            video_frame_local = video_frames[i*batch:(i+1)*batch]
            if len(video_frame_local) < 1:
                break
            
            detected_faces_list = self.face_detector(video_frame_local, rgb=False)
            if len(detected_faces_list) != len(video_frame_local):
                detected_faces_list = self.face_detector(video_frame_local, rgb=False, old=True)
            for info in detected_faces_list:
                if len(info) > 0:
                    landmarks.append(np.array(np.round(info[0, 5:]), dtype=int))
                else:
                    landmarks.append(np.full(10, -1)) 
        return landmarks
