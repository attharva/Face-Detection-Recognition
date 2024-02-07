'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import cv2
import numpy as np
import os
import sys
import math

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: np.ndarray) -> List[List[float]]:
    """
    Args:
        img : input image is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """

    detect = face_recognition.face_locations(img)

    conv = []
    for i in detect:
        x_left = i[3]
        h = i[2] - i[0]
        w = i[1] - i[3]
        y_left = i[0]
        conv.append([float(x_left), float(y_left), float(w), float(h)])
    # print(conv)
    return conv


def cluster_faces(imgs: Dict[str, np.ndarray], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    cluster_results: List[List[str]] = [[]] * K # Please make sure your output follows this data format.

    
    encoded_images = [face_recognition.api.face_encodings(imgs[x])[0] for x in imgs ]

    image_map = [x for x in imgs.keys()]

    np.random.seed(1000)
    centers = [encoded_images[x] for x  in np.random.choice(range(len(encoded_images)),size = K)]
    cluster_groups =[]
    loop = 0
    
    while loop <= 10000:
        cluster = {}

        for index in range(len(encoded_images)):
            errors = {}
            for cnt in range(K):
                err = math.dist(centers[cnt],encoded_images[index])
                errors[cnt] = err

            value_list = list(errors.values())

            cluster[index]  = value_list.index(min(value_list))
        
        temp = centers

        cluster_0 =[]
        cluster_2 =[]
        cluster_3 =[]
        cluster_4 =[]
        cluster_1 =[]

        for k,i in zip(list(cluster.values()), range(len(cluster.values()))):
            if k ==0:
                cluster_0.append(encoded_images[i])
            if k ==1:
                cluster_1.append(encoded_images[i])
            if k == 2:
                cluster_2.append(encoded_images[i])
            if k ==3:
                cluster_3.append(encoded_images[i])
            if k ==4:
                cluster_4.append(encoded_images[i])
        
        centers =[]
        centers.append(np.mean(np.asarray(cluster_0),axis =0))
        centers.append(np.mean(np.asarray(cluster_1),axis =0))
        centers.append(np.mean(np.asarray(cluster_2),axis =0))
        centers.append(np.mean(np.asarray(cluster_3),axis =0))
        centers.append(np.mean(np.asarray(cluster_4),axis =0))

        if loop >0:
            if np.array_equal(np.asarray(temp), np.asarray(centers)):
                cluster_groups.append(cluster_0)
                cluster_groups.append(cluster_1)
                cluster_groups.append(cluster_2)
                cluster_groups.append(cluster_3)
                cluster_groups.append(cluster_4)
                break
                
        loop = loop+ 1
    final =[]
    for group in cluster_groups:
        name_list =[]
        for i in group:
            for code_index in range(len(encoded_images)):
                if np.array_equal(encoded_images[code_index],i):
                    name = image_map[code_index]
            name_list.append(name)
        final.append(name_list)
    
    return final


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# Your functions. (if needed)
