import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import random

def convert_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def read_image(path):
    return mpimg.imread(path)

def flip_img(img):
    return np.fliplr(img)

def flip_angle(angle):
    return -angle

def crop_img(img):
    return img[50:140, : ]

def resize_img(img):
    return cv2.resize(img, (200, 66))

def random_contrast(img):
    img_temp = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_temp[:,:,2] = img_temp[:,:,2] * np.random.uniform(.4, 1.)
    img_temp = cv2.cvtColor(img_temp, cv2.COLOR_HSV2RGB)
    return img_temp

def jitter_transition(img, angle):
    transRange = 50
    numPix = 10
    valPix = 0.1
    transX = transRange * np.random.uniform() - transRange / 2
    angle = angle + transX / transRange * 2 * valPix
    transY = numPix * np.random.uniform() - numPix / 2
    transMat = np.float32([[1, 0, transX], [0, 1, transY]])
    img = cv2.warpAffine(img, transMat, (200, 66))
    
    return img, angle

def transition_image(img, angle):
    trans_val_x = .50
    trans_val_y = .015
    trans_x = trans_val_x * np.random.uniform() - trans_val_x / 2
    angle = angle + trans_x / trans_val_x * 2 * 0.2
    trans_y = trans_val_y * np.random.uniform() - trans_val_y / 2
    trans_mat = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    img = cv2.warpAffine(img, trans_mat,(200,66))
    
    return img, angle

def select_camera(path):
    camera_selection = [0, 1, 2]
#     camera_selection = [1]
    camera_choice = random.choice(camera_selection)
    if camera_choice == 0:
        camera_file = path['left']
        ang = path['steering'][0] + 0.25
    elif camera_choice == 1:
        camera_file = path['center']
        ang = path['steering'][0]
    elif camera_choice == 2:
        camera_file = path['right']
        ang = path['steering'][0] - 0.25
        
    return camera_file, ang
    
def preprocess_train(img, ang):
    img = crop_img(img)
    img = resize_img(img)
    img = random_contrast(img)
#     img, ang = transition_image(img, ang)
    img, ang = jitter_transition(img, ang)
    flip_choice = random.choice([0, 1])
    if flip_choice == 1:
        img = flip_img(img)
        ang = flip_angle(ang)
    else:
        img = img
        ang = ang
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, ang

def preprocess_valid(img, ang):
    img = crop_img(img)
    img = resize_img(img)
    img = random_contrast(img)
    flip_choice = random.choice([0, 1])
    if flip_choice == 1:
        img = flip_img(img)
        ang = flip_angle(ang)
    return img, ang

def train_pipe(path, batch_size = 250):
    csv_data = pd.read_csv(path + '/driving_log.csv')
    
#     batch_images = np.empty((batch_size, 66, 200, 3))
#     batch_angles = np.empty(batch_size)
    while True:
        batch_images = []
        batch_angles = []
        batch_full = False
        batch_count = 0
        actual_count = 0
#         for i in range(batch_size):
        while batch_full == False:
            
            line_loc = np.random.randint(0, len(csv_data))# - i)
            line_data = csv_data.iloc[[line_loc]].reset_index()

            image_file, angle = select_camera(line_data)
            image_file = image_file[0]

            final_path = path + '/IMG/' + image_file.strip()

            image = read_image(final_path)


            image, angle = preprocess_train(image, angle)


#             if abs(angle) < 0.15:
#                 if np.random.uniform() > 0.1:
#                     batch_images[batch_count] = image
#                     batch_angles[batch_count] = angle
#                     batch_count += 1
#             else:
#                 batch_images[batch_count] = image
#                 batch_angles[batch_count] = angle
#                 batch_count += 1
            
#             batch_images[batch_count] = image
#             batch_angles[batch_count] = angle
            if abs(angle) <= 0.25:
                if np.random.uniform() > 0.9:
                    batch_images.append(image)
                    batch_angles.append(angle)
                    batch_count += 1
            elif ((abs(angle) > 0.25) & (abs(angle) <= 0.5)):
                if np.random.uniform() > 0.9:
                    batch_images.append(image)
                    batch_angles.append(angle)
                    batch_count += 1
            elif ((abs(angle) > 0.5) & (abs(angle) <= 0.75)):
                if np.random.uniform() > 0.2:
                    batch_images.append(image)
                    batch_angles.append(angle)
                    batch_count += 1
            elif ((abs(angle) > 0.75) & (abs(angle) <= 1.0)):
                if np.random.uniform() > 0:
                    batch_images.append(image)
                    batch_angles.append(angle)
                    batch_count += 1
            else:
                pass
#                 batch_images.append(image)
#                 batch_angles.append(angle)
#                 batch_count += 1
#             batch_images.append(image)
#             batch_angles.append(angle)
#             batch_count += 1
            
            if batch_count == batch_size:
                batch_full = True
            actual_count += 1
#             csv_data.drop(csv_data.index[line_loc])
            

        yield np.array(batch_images), np.array(batch_angles)
            


def valid_pipe(path):
    csv_data = pd.read_csv(path + '/driving_log.csv')
    
    while True:
        for i in range(len(csv_data)):
            line_data = csv_data.iloc[[i]].reset_index()
            image_path = line_data['center'][0].strip()
            final_path = path + '/IMG/' + image_path
            image = read_image(final_path)
            angle = line_data['steering'][0]
            
            image, angle = preprocess_valid(image, angle)
            
            image = image.reshape(1, 66, 200, 3)
            angle = np.array([[angle]])
            yield image, angle