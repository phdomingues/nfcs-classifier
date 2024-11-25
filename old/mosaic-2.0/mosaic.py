import cv2
import math
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import urllib.request

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarkerResult
from mediapipe.tasks.python import vision
from typing import List, Tuple

class FaceMosaic:
    def __init__(self, detection=False) -> None:
        # Detection + landmarks
        # if detection:
        model_path, http_msg = urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task", 
            "face_landmarker.task")
        self.options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5)
        # # Landmarks only - https://github.com/googlesamples/mediapipe/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb?source=post_page-----6381dbf78756--------------------------------
        # else:
        #     base_options = BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')        
        #     self.options = vision.FaceLandmarkerOptions(base_options=base_options,
        #                                output_face_blendshapes=True,
        #                                output_facial_transformation_matrixes=True,
        #                                num_faces=1)

        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.regions = { # Reference: https://i.stack.imgur.com/wDgvV.png
            'mouth': [
                [57, 43, 106, 182, 83, 18, 313, 406, 335, 273, 287, 410, 322, 391, 393, 164, 167, 165, 92, 186]
                # [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185] # Tight
                ],
            'palpebral fissure': [
                [226, 31, 228, 229, 230, 231, 232, 233, 244, 245, 122, 6, 351, 465, 464, 453, 452, 451, 450, 449, 448, 261, 446, 342, 445, 444, 443, 442, 441, 417, 168, 193, 221, 222, 223, 224, 225, 113]
                # [463, 341, 256, 252, 253, 254, 339, 255, 359, 467, 260, 259, 257, 258, 286, 414], # Tight Left
                # [243, 112, 26, 22, 23, 24, 110, 25, 130, 247, 30, 29, 27, 28, 56, 190]  # Tight Right

            ],
            'nasolabial fold': [
                [2, 97, 98, 92, 61, 43, 204, 211, 170, 149, 150, 136, 135, 214, 207, 205, 36, 142, 126, 217, 174, 196, 197,
                 419, 399, 437, 355, 371, 266, 425, 427, 434, 364, 365, 379, 378, 395, 431, 424, 273, 291, 322, 327, 326]
                # [287, 432, 436, 426, 423, 358, 327, 391, 322, 410], # Tight Left
                # [57, 212, 216, 206, 203, 129, 98, 165, 92, 186]  # Tight Right
            ],
            'forehead': [
                [168, 417, 285, 336, 296, 334, 293, 300, 298, 332, 297, 338, 10, 109, 67, 103, 68, 70, 63, 105, 66, 107, 55, 193]]
        }

    def _normalized2pix_coords(self, normalized_x, normalized_y, img_shape):
        x_px = min(math.floor(normalized_x * img_shape[1]), img_shape[1] - 1)
        y_px = min(math.floor(normalized_y * img_shape[0]), img_shape[0] - 1)
        return (x_px, y_px)

    def settings_from_shape(self, img_shape):
        return {
            'thickness': (img_shape[0]*img_shape[1])//2_000_000 + 1,
            'circle_radius': (img_shape[0]*img_shape[1])//1_500_000 + 1
        }
    
    def draw_landmarks(self, detection_result, img):
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            display_settings = self.settings_from_shape(img.shape)

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=solutions.drawing_utils.DrawingSpec(thickness=display_settings['thickness'], circle_radius=display_settings['circle_radius']),
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()#solutions.drawing_utils.DrawingSpec(thickness=display_settings['thickness'], color=(255,0,0))#mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                )
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=solutions.drawing_utils.DrawingSpec(thickness=display_settings['thickness'], circle_radius=display_settings['circle_radius']),
                connection_drawing_spec=solutions.drawing_utils.DrawingSpec(thickness=display_settings['thickness'], color=(0,255,0))#mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
                )
            #solutions.drawing_utils.draw_landmarks(
                #image=annotated_image,
                #landmark_list=face_landmarks_proto,
                #connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                #landmark_drawing_spec=solutions.drawing_utils.DrawingSpec(thickness=display_settings['thickness'], circle_radius=display_settings['circle_radius']),
                #connection_drawing_spec=solutions.drawing_utils.DrawingSpec(thickness=display_settings['thickness'], color=(0,0,255))#mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
                #)

        return annotated_image

    def draw_mosaic(self, detection_result, img, points=True, lines=True, fill=True):
        display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        display_settings = self.settings_from_shape(display_img.shape)
        # overlay = np.zeros(display_img.shape)
        for face in detection_result.face_landmarks:
            for region in self.regions.values():
                for sub_region in region:
                    coords = [self._normalized2pix_coords(face[mask_coord].x, face[mask_coord].y, display_img.shape) for mask_coord in sub_region]
                    for i, coord in enumerate(coords):
                        # Next dot in order
                        next_idx = (i+1)%len(coords)
                        next_coord = coords[next_idx]
                        # Paint image
                        if points:
                            display_img = cv2.circle(display_img, 
                                coord, # Center coordinate
                                display_settings['circle_radius'], # Radius
                                (0, 0, 255), # Color
                                -1) # Thickness (-1 for fill)
                        if lines:
                            display_img = cv2.line(display_img,
                                coord, # Start point
                                next_coord, # End point
                                (0, 255, 0), # Color
                                display_settings['thickness'])
                    if fill:
                        overlay= display_img.copy()
                        overlay = cv2.fillPoly(overlay, # Image
                            np.array([coords], dtype=np.int32), # Polygon vertices
                            (255,0,0), # Color
                            cv2.LINE_AA)  # Line type
                        display_img = cv2.addWeighted(overlay, 0.4, display_img, 1 - 0.4, 0)#, dtype=cv2.CV_64F)

        return display_img

    def create_masks(self, detection_result, img):
        masks = {region: np.zeros((img.shape[0], img.shape[1])) for region in self.regions}
        for face in detection_result.face_landmarks:
            for region, region_data in self.regions.items():
                for sub_region in region_data:
                    coords = [self._normalized2pix_coords(face[mask_coord].x, face[mask_coord].y, img.shape) for mask_coord in sub_region]
                    masks[region] = cv2.fillPoly(masks[region], # Image
                        np.array([coords], dtype=np.int32), # Polygon vertices
                        255, # Color
                        cv2.LINE_AA)  # Line type
                    masks[region] = masks[region].astype(np.int8)
        return masks

    def apply_mask(self, img, mask, crop=False):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
        if crop:
            masked_img = self.crop(masked_img, mask)
        return masked_img

    def crop(self, img, mask):
        rows, cols = np.where(np.abs(mask)>0)
        final_img = img.copy()[min(rows):max(rows)+1, min(cols):max(cols)+1]
        return final_img
    
    def detect_landmarks(self, img_path:List[str]|str, return_img:bool=True) -> List[Tuple[FaceLandmarkerResult, mp.Image]]:
        if isinstance(img_path, str):
            img_path = [img_path]
        results = []
        with self.FaceLandmarker.create_from_options(self.options) as landmarker:
            for ip in img_path:
                # Load the input image from a numpy array.
                numpy_image = cv2.imread(ip)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
                # Perform face landmarking on the provided single image.
                # The face landmarker must be created with the image mode.
                face_landmarker_result = landmarker.detect(mp_image)

                results.append((face_landmarker_result, numpy_image if return_img else None))
        
        return results
