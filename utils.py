from __future__ import print_function

import torch
import pickle
import numpy as np
import math
import cv2
from PIL import Image
import hashlib
import sys, os
from zipfile import ZipFile
from skimage import io

if sys.version_info[0] == 2:
    from six.moves.urllib.request import urlretrieve
else:
    from urllib.request import urlretrieve
def diff(input_list):
    return max(input_list)-min(input_list)
def loadImage(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)
    height,width,b = img.shape
    if width > 1000 and height >1000:
        ratio = width/height
        img = cv2.resize(img, (int(800*ratio), 800), interpolation =  Image.ANTIALIAS)

    return img
def printProgressBar (prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    def progress_hook(count, blockSize, totalSize):
        progress = count * blockSize / totalSize
        percent = ("{0:." + str(decimals) + "f}").format(progress * 100)
        filledLength = int(length * progress)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)

    return progress_hook
def reformat_input(image):
    if type(image) == str:
        if image.startswith('http://') or image.startswith('https://'):
            tmp, _ = urlretrieve(image , reporthook=printProgressBar(prefix = 'Progress:', suffix = 'Complete', length = 50))
            img_cv_grey = cv2.imread(tmp, cv2.IMREAD_GRAYSCALE)
            os.remove(tmp)
        else:
            img_cv_grey = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            image = os.path.expanduser(image)
        img = loadImage(image)  # can accept URL
    elif type(image) == bytes:
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elif type(image) == np.ndarray:
        if len(image.shape) == 2: # grayscale
            img_cv_grey = image
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 3: # BGRscale
            img = image
            img_cv_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4: # RGBAscale
            img = image[:,:,:3]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        LOGGER.warning('Invalid input type. Suppoting format = string(file path or url), bytes, numpy array')

    return img, img_cv_grey

def group_text_box(polys, slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5, width_ths = 1.0, add_margin = 0.05):
    # poly top-left, top-right, low-right, low-left
    horizontal_list, free_list,combined_list, merged_list = [],[],[],[]

    for poly in polys:
        slope_up = (poly[3]-poly[1])/np.maximum(10, (poly[2]-poly[0]))
        slope_down = (poly[5]-poly[7])/np.maximum(10, (poly[4]-poly[6]))
        if max(abs(slope_up), abs(slope_down)) < slope_ths:
            x_max = max([poly[0],poly[2],poly[4],poly[6]])
            x_min = min([poly[0],poly[2],poly[4],poly[6]])
            y_max = max([poly[1],poly[3],poly[5],poly[7]])
            y_min = min([poly[1],poly[3],poly[5],poly[7]])
            horizontal_list.append([x_min, x_max, y_min, y_max, 0.5*(y_min+y_max), y_max-y_min])
        else:
            height = np.linalg.norm( [poly[6]-poly[0],poly[7]-poly[1]])
            margin = int(1.44*add_margin*height)

            theta13 = abs(np.arctan( (poly[1]-poly[5])/np.maximum(10, (poly[0]-poly[4]))))
            theta24 = abs(np.arctan( (poly[3]-poly[7])/np.maximum(10, (poly[2]-poly[6]))))
            # do I need to clip minimum, maximum value here?
            x1 = poly[0] - np.cos(theta13)*margin
            y1 = poly[1] - np.sin(theta13)*margin
            x2 = poly[2] + np.cos(theta24)*margin
            y2 = poly[3] - np.sin(theta24)*margin
            x3 = poly[4] + np.cos(theta13)*margin
            y3 = poly[5] + np.sin(theta13)*margin
            x4 = poly[6] - np.cos(theta24)*margin
            y4 = poly[7] + np.sin(theta24)*margin

            free_list.append([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
    horizontal_list = sorted(horizontal_list, key=lambda item: item[4])

    # combine box
    new_box = []
    for poly in horizontal_list:

        if len(new_box) == 0:
            b_height = [poly[5]]
            b_ycenter = [poly[4]]
            new_box.append(poly)
        else:
            # comparable height and comparable y_center level up to ths*height
            if (abs(np.mean(b_height) - poly[5]) < height_ths*np.mean(b_height)) and (abs(np.mean(b_ycenter) - poly[4]) < ycenter_ths*np.mean(b_height)):
                b_height.append(poly[5])
                b_ycenter.append(poly[4])
                new_box.append(poly)
            else:
                b_height = [poly[5]]
                b_ycenter = [poly[4]]
                combined_list.append(new_box)
                new_box = [poly]
    combined_list.append(new_box)

    # merge list use sort again
    for boxes in combined_list:
        if len(boxes) == 1: # one box per line
            box = boxes[0]
            margin = int(add_margin*box[5])
            merged_list.append([box[0]-margin,box[1]+margin,box[2]-margin,box[3]+margin])
        else: # multiple boxes per line
            boxes = sorted(boxes, key=lambda item: item[0])

            merged_box, new_box = [],[]
            for box in boxes:
                if len(new_box) == 0:
                    x_max = box[1]
                    new_box.append(box)
                else:
                    if abs(box[0]-x_max) < width_ths *(box[3]-box[2]): # merge boxes
                        x_max = box[1]
                        new_box.append(box)
                    else:
                        x_max = box[1]
                        merged_box.append(new_box)
                        new_box = [box]
            if len(new_box) >0: merged_box.append(new_box)

            for mbox in merged_box:
                if len(mbox) != 1: # adjacent box in same line
                    # do I need to add margin here?
                    x_min = min(mbox, key=lambda x: x[0])[0]
                    x_max = max(mbox, key=lambda x: x[1])[1]
                    y_min = min(mbox, key=lambda x: x[2])[2]
                    y_max = max(mbox, key=lambda x: x[3])[3]

                    margin = int(add_margin*(y_max - y_min))

                    merged_list.append([x_min-margin, x_max+margin, y_min-margin, y_max+margin])
                else: # non adjacent box in same line
                    box = mbox[0]

                    margin = int(add_margin*(box[3] - box[2]))
                    merged_list.append([box[0]-margin,box[1]+margin,box[2]-margin,box[3]+margin])
    # may need to check if box is really in image
    return merged_list, free_list
def four_point_transform(image, rect):
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped
def get_image_list(horizontal_list, free_list, img, model_height = 32):
    image_list = []

    maximum_y,maximum_x,b = img.shape

    max_ratio_hori, max_ratio_free = 1,1
    for box in free_list:
        rect = np.array(box, dtype = "float32")
        transformed_img = four_point_transform(img, rect)
        ratio = transformed_img.shape[1]/transformed_img.shape[0]
        crop_img = cv2.resize(transformed_img, (int(model_height*ratio), model_height), interpolation =  Image.ANTIALIAS)
        image_list.append( (box,crop_img) ) # box = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        max_ratio_free = max(ratio, max_ratio_free)

    max_ratio_free = math.ceil(max_ratio_free)
    for box in horizontal_list:
        x_min = max(0,box[0])
        x_max = min(box[1],maximum_x)
        y_min = max(0,box[2])
        y_max = min(box[3],maximum_y)
        crop_img = img[y_min : y_max, x_min:x_max,:]
        width = x_max - x_min
        height = y_max - y_min
        ratio = width/height
        crop_img = cv2.resize(crop_img, (int(model_height*ratio), model_height), interpolation =  Image.ANTIALIAS)
        image_list.append( ( [[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]] ,crop_img) )
        max_ratio_hori = max(ratio, max_ratio_hori)

    max_ratio_hori = math.ceil(max_ratio_hori)
    max_ratio = max(max_ratio_hori, max_ratio_free)
    max_width = math.ceil(max_ratio)*model_height

    image_list = sorted(image_list, key=lambda item: item[0][0][1]) # sort by vertical position
    return image_list, max_width
