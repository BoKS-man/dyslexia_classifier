import os
import cv2 as cv
import numpy as np
import pandas as pd
from scipy import stats as st
from torch.utils.data import Dataset, DataLoader

def crop_img(img:np.ndarray|list[int], crop_size_h_w:list[int]=[224], step_x_y:list[int]=[40]):
    crops = []
    px_means = []
    crop_coords = []
    if not (len(crop_size_h_w) > 0 and len(crop_size_h_w) <= 2):
        raise ValueError(f'crop_size_h_w must be provided with one ot two values, but get {len(crop_size_h_w)}')
    crop_h, crop_w = crop_size_h_w if len(crop_size_h_w) == 2 else 2*crop_size_h_w
    if not (len(step_x_y) > 0 and len(step_x_y) <= 2):
        raise ValueError(f'step_x_y must be provided with one ot two values, but get {len(step_x_y)}')
    step_x, step_y = step_x_y if len(step_x_y) == 2 else 2*step_x_y
    img_h, img_w, _ = img.shape
    x, y = (0, 0)
    while img_h - y >= crop_h:
        while img_w - x >= crop_w:
            crop = img[y:y+crop_w, x:x+crop_h]
            px_means.append(crop.mean())
            crop_coords.append((x, y))
            crops.append(crop)
            x += step_x
        y += step_y
        x = 0
    max_mean = st.norm.interval(alpha=.95, loc=np.mean(px_means), scale=st.sem(px_means))[1]
    f_idx = np.where(px_means <= max_mean)[0]
    crops = np.asarray(crops)[f_idx]
    px_means = np.asarray(px_means)[f_idx]
    crop_coords = np.asarray(crop_coords)[f_idx]

    return crops, crop_coords, px_means

def get_datasets(dys_path:str, not_dys_path:str, \
                 crop_size_h_w:list[int]=[224], crop_step_x_y:list[int]=[40], \
                 val:bool=False, test_val_part:list[float]=[.15]):
    for data_path in [dys_path, not_dys_path]:
        if not os.path.exists(data_path):
            raise ValueError(f'path {data_path} not exists')
    y = []
    crops = []
    sources = []
    crop_coords = []
    crop_px_means = []
    for data_path, dys in zip([dys_path, not_dys_path], [True, False]):
        files = [os.path.join(t[0], f) for t in os.walk(data_path) for f in t[2] if f.endswith('jpg')]
        for file in files:
            img = cv.imread(file)
            img_crops, img_crop_coords, img_crop_px_means = crop_img(img, crop_size_h_w, crop_step_x_y)
            crops += list(img_crops)
            y += len(img_crops)*[dys]
            crop_coords += list(img_crop_coords)
            sources += len(img_crops)*[file]
            crop_px_means += list(img_crop_px_means)
    #print(crop_coords[:5])
    #print(crop_px_means[:5])
    #print(sources[:5])
    #print(y[:5])
    data = np.concatenate((np.asarray(crop_coords),
                           np.expand_dims(crop_px_means, axis=1),
                           np.expand_dims(sources, axis=1), 
                           np.expand_dims(y, axis=1)), axis=1)
    #print(data.shape)
    data = pd.DataFrame(data, columns=['coord_x', 'coord_y', 'px_mean', 'source', 'y'])
    return data

