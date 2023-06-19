import os
import cv2 as cv
import numpy as np
import pandas as pd
from scipy import stats as st
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

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

def get_datasets(dys_path:str, not_dys_path:str, batch_size:int=1, \
                 crop_size_h_w:list[int]=[224], crop_step_x_y:list[int]=[40], \
                 val:bool=False, test_val_part:list[float]=[.15]):
    for data_path in [dys_path, not_dys_path]:
        if not os.path.exists(data_path):
            raise ValueError(f'path {data_path} not exists')
    if not val and len(test_val_part) > 1:
        test_val_part = test_val_part[0]
    if test_val_part == 0: 
        raise ValueError('test_val_part must be specified')
    if val and len(test_val_part) == 1:
        test_val_part = 2*test_val_part
    if val and len(test_val_part) > 2:
        test_val_part = test_val_part[:2]
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
    crops = np.asarray(crops)
    data = list(zip(crop_coords, crop_px_means, sources, y))
    data = pd.DataFrame(data, columns=['coords', 'px_mean', 'source', 'y'])
    data.reset_index(inplace=True)
    data['int_mean'] = 0
    px_mean_borders = [0] + [np.percentile(data['px_mean'], p*10) for p in list(range(1, 10))] + [data['px_mean'].max()]
    for i in range(10):
        data.loc[list(data[(data['px_mean']>px_mean_borders[i]) & (data['px_mean']<=px_mean_borders[i+1])].index), 'int_mean'] = i
    train, test = train_test_split(data, test_size=sum(test_val_part), stratify=data[['y', 'int_mean']])
    if val:
        test, val = train_test_split(test, test_size=np.divide(*test_val_part)/2, stratify=test[['y', 'int_mean']])
    #return train, test
    return [None if type(ds) != pd.DataFrame else \
            DataLoader(DysDataset(ds, crops[list(ds['index'].values)]), batch_size, True, num_workers=1) \
                for ds in [train, test, val]]

class DysDataset(Dataset):
    def __init__(self, data, imgs):
        self.__imgs = imgs
        self.__y = data['y'].values

    def __getitem__(self, i):
        return self.__imgs[i], self.__y[i]
    
    def __len__(self):
        return len(self.__y)