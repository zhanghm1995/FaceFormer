"""Detect faces in every image
"""
import numpy as np
import cv2
import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import argparse
import face_alignment
import pickle
import torch

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, face_detector='sfd')

def detect_face_from_one_image(input_image):
    """Detect one face from a single input image, we assume there not more than 1 face in this image

    Args:
        input_image (cv2 image): Image type

    Returns:
        list: [x1,y1,x2,y2] to represent the face coordinate in image, return None if don't detect any faces
    """
    detected_faces = fa.face_detector.detect_from_image(input_image.copy())

    if len(detected_faces) == 0:
        return None

    face = detected_faces[0] # [x1,y1,x2,y2,confidence]
    return face[:-1]


def detect_face_from_batch_image(image_list):
    images = np.array(image_list)
    images = images[..., ::-1].transpose(0,3,1,2)
    images = torch.from_numpy(images.copy())

    detected_faces = fa.face_detector.detect_from_batch(images)

    results = []

    for i, d in enumerate(detected_faces):
        if len(d) == 0:
            results.append(None)
            continue
        d = d[0]
        d = np.clip(d, 0, None)
        
        x1, y1, x2, y2 = map(int, d[:-1])
        results.append((x1, y1, x2, y2))

    return results


def detect_face(input_image, resize_factor=1):
    if resize_factor > 1:
        frame_resized = cv2.resize(input_image, (input_image.shape[1]//resize_factor, input_image.shape[0]//resize_factor))
        face_coord = detect_face_from_one_image(frame_resized)
    else:
        face_coord = detect_face_from_one_image(input_image)

    if face_coord is None:
       return None
    
    if resize_factor > 1:
        face_coord = tuple(resize_factor * x for x in face_coord)

    face_coord = np.clip(face_coord, 0, None)
    return face_coord
    

def detect_landmarks(input_image, return_bboxes=False, resize_factor=1):
    if resize_factor > 1:
        input_image = cv2.resize(input_image, (input_image.shape[1]//resize_factor, input_image.shape[0]//resize_factor))

    detect_results = fa.get_landmarks_from_image(input_image, return_bboxes=return_bboxes)

    if return_bboxes:
        landmarks, _, detected_faces = detect_results
        landmarks = [resize_factor * x for x in landmarks]
        detected_faces = [resize_factor * x[:-1] for x in detected_faces]

        detected_faces = np.clip(detected_faces, 0, None)
        
        if len(landmarks) == 0:
            return None

        return landmarks, detected_faces
    else:
        landmarks = detect_results

        if len(landmarks) == 0:
            return None
        return landmarks
    