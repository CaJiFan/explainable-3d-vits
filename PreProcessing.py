#!/usr/bin/env python
# coding: utf-8

import shutil
import difflib
import numpy as np
import SimpleITK as sitk
import scipy.spatial
from glob import glob
from typing import  List
from pprint import pprint
import matplotlib.pyplot as plt
from skimage.transform import resize
import os
import time
import random
import warnings
import scipy
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from utils import *
import ants
from antspynet.utilities import preprocess_brain_image
from sklearn.model_selection import train_test_split

print(tf.keras.__version__)
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

base_data_path = '/home/carlos.jimenez/ai-in-health/ICD-Transformers/data-edges'
# base_data_path = '/home/carlos.jimenez/data'

seed = 1337

icd_data_paths = sorted(glob(f'{base_data_path}/ICD/nifty/*brain_extracted*edges.nii.gz'))
icd_data_paths, _ = train_test_split(icd_data_paths, test_size=0.25,random_state=seed)

apathy_icd_data_paths = sorted(glob(f'{base_data_path}/apathy+ICD/nifty/*brain_extracted*edges.nii.gz'))
print(len(apathy_icd_data_paths))

apathy_data_paths = sorted(glob(f'{base_data_path}/apathy/nifty/*brain_extracted*edges.nii.gz'))
print(len(apathy_data_paths))

pd_only_data_paths = sorted(glob(f'{base_data_path}/PD_Only/nifty/*brain_extracted*edges.nii.gz'))[:8208]
print(len(pd_only_data_paths))
pd_only_data_paths, _ = train_test_split(pd_only_data_paths, test_size=0.80,random_state=seed)


data_paths = icd_data_paths + apathy_icd_data_paths + apathy_data_paths + pd_only_data_paths

print(f'ICD: {len(icd_data_paths)}')
print(f'Apathy+ICD: {len(apathy_icd_data_paths)}')
print(f'Apathy only: {len(apathy_data_paths)}')
print(f'PD only: {len(pd_only_data_paths)}')
print('Subtotal', len(data_paths))

data_paths = [path for path in data_paths if not ('contrast' in path)]
print(f'Total after removing contrast: {len(data_paths)}')
# data_paths = [path for path in data_paths if not ('edges' in path  or 'contrast' in path or 'denoised_brain_extracted_rotated' in path)]


# ## Loading MNI T1 Template
mni_T1_path = f'{base_data_path}/templates/mni_icbm152_t1_tal_nlin_sym_09a.nii'
mni_T1_ANTSPY = ants.image_read(mni_T1_path)
direction = mni_T1_ANTSPY.direction

mni_sitk = sitk.ReadImage(mni_T1_path, sitk.sitkFloat32)
plot_3d_overview(sitk.GetArrayFromImage(mni_sitk))


# Paths and constants
BASE_DATA_PATH = "/home/carlos.jimenez/ai-in-health/ICD-Transformers/og-data/data"
classes_dict = {
    'PD_Only': 0,
    'apathy': 1,
    'ICD': 2,
    'apathy+ICD': 3,
}

# Filtrar paths y excluir archivos con "registered"
paths = []
for class_name in classes_dict.keys():
    paths.extend(sorted(glob(f'{BASE_DATA_PATH}/{class_name}/nifty/*.nii.gz')))
filtered_paths = [path for path in paths if "registered" not in os.path.basename(path)]


# ## Getting Data paths

registered_paths = sorted(glob(f'{base_data_path}/*/nifty/*.nii.gz'))
registered_brain_extracted_paths = sorted(glob(f'{base_data_path}/*/nifty/*brain_extracted*.nii.gz'))
print(registered_paths[0])
print(len(registered_brain_extracted_paths))

mni_sitk = sitk.ReadImage(registered_paths[0], sitk.sitkFloat32)
plot_3d_overview(sitk.GetArrayFromImage(mni_sitk))





icd_xpaths = sorted(glob(f'{base_data_path}/ICD/nifty/*brain_extracted*edges.nii.gz'))
apathy_xpaths = sorted(glob(f'{base_data_path}/apathy/nifty/*brain_extracted*edges.nii.gz'))
apathy_icd_xpaths = sorted(glob(f'{base_data_path}/apathy+ICD/nifty/*brain_extracted*edges.nii.gz'))
pd_only_xpaths = sorted(glob(f'{base_data_path}/PD_Only/nifty/*brain_extracted*edges.nii.gz'))
print(len(icd_xpaths))
print(len(apathy_xpaths))
print(len(apathy_icd_xpaths))
print(len(pd_only_xpaths))


# In[ ]:


for path in pd_only_xpaths[:3]:
    img = sitk.ReadImage(path, sitk.sitkFloat32)
    plot_3d_overview(sitk.GetArrayFromImage(img))


# In[21]:


print(mni_sitk.GetOrigin())


# ## Normalizing images

# In[37]:


for i,example in enumerate(pd_only_xpaths+icd_xpaths+apathy_xpaths+apathy_icd_xpaths):  
    img = sitk.ReadImage(example, sitk.sitkFloat32)
    origin = img.GetOrigin() == mni_sitk.GetOrigin()
    direction = img.GetDirection() == mni_sitk.GetDirection()
    spacing = img.GetSpacing() == mni_sitk.GetSpacing()
    
    ov_np = sitk.GetArrayFromImage(img)
    shape = ov_np.shape == (189, 233, 197)
    max_val = ov_np.max()
    
    if not direction:
        print(i, example)
        print('direction',img.GetDirection())
    if not origin:
        print(i, example)
        print('origin', img.GetOrigin())
    if not spacing:
        print(i, example)
        print('shape', ov_np.shape)
    if not shape:
        print(i, example)
        print('shape', ov_np.shape)
    if max_val != 1.0:
        print(i, example)
        print('max', max_val)


# In[34]:


for i, attn_map_path in enumerate(pd_only_xpaths+icd_xpaths+apathy_xpaths+apathy_icd_xpaths):
    attn_map = sitk.ReadImage(attn_map_path, sitk.sitkFloat32)
    attn_map_np = sitk.GetArrayFromImage(attn_map)
    
    if attn_map_np.max() != 1.0:
        print(i, attn_map_path)
        # min max normalization
#         min_v, max_v = attn_map_np.min(), attn_map_np.max()
#         attn_map_np = (attn_map_np - min_v) / (max_v - min_v)

#         # saving image
#         corrected_brain_extracted = sitk.GetImageFromArray(attn_map_np)
#         corrected_brain_extracted.CopyInformation(mni_sitk)

#         sitk.WriteImage(corrected_brain_extracted, attn_map_path)


# In[28]:


for i,example in enumerate(icd_xpaths):  
    img = sitk.ReadImage(example, sitk.sitkFloat32)
    origin = img.GetOrigin() == mni_sitk.GetOrigin()
#     print(img.GetOrigin())
#     direction = img.GetOrigin() == (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
#     print(img)
    if not origin:
        print(i, example)
        img.SetOrigin(mni_sitk.GetOrigin())
        sitk.WriteImage(img, example)
        
#     origin = img.GetOrigin() == (98.0, 134.0, -72.0)
#     spacing = img.GetSpacing() == (1.0, 1.0, 1.0)
    
#     ov_np = sitk.GetArrayFromImage(img)
#     shape = ov_np.shape == (189, 233, 197)
    
#     if not direction or not origin or not spacing or not shape:
#         print(i, example)
#         print('direction',img.GetDirection())
#         print('origin', img.GetOrigin())
#         print('spacing', img.GetSpacing())
#         print('shape', ov_np.shape)
#         max_val = ov_np.max()
#         print(max_val)


# In[ ]:





# In[26]:


# pd_only_xpaths


# In[17]:


import os

for path in pd_only_xpaths:
    if not path.endswith('edges.nii.gz'):
#         print(path)
        term = '_edges_'
        edges_idx = path.index(term)
        new_path = f'{path[:edges_idx]}_{path[edges_idx+len(term):-7]}_edges.nii.gz'
#         print(new_path)
#         print()
        os.rename(path, new_path)
    


# In[ ]:


xpath = registered_brain_extracted_paths[1]
print(xpath)
ximg = ants.image_read(xpath)


# ## Complete T1 Image Preprocessing Pipeline

# In[ ]:


# ximg = ants.image_read(icd_xpaths[0])
ximg2 = ants.image_read(icd_xpaths[0][:-18] + '.nii.gz')
preprocessed = preprocess_brain_image(
                ximg2,
                truncate_intensity=None,
                brain_extraction_modality='t1',
                template_transform_type='TRSAA',
                template=mni_T1_ANTSPY,
                do_bias_correction=True,
                do_denoising=False,
                intensity_normalization_type='01',
                verbose=False
            )
preprocessed_image = preprocessed['preprocessed_image']
ants.plot(preprocessed_image)


# In[11]:


for i, path in enumerate(pd_only_xpaths):
    ximg = ants.image_read(path)
    file_name = path[:-7]
    x_registered_path = file_name + '_registered.nii.gz'
    x_brain_extracted_path = file_name + '_registered_brain_extracted.nii.gz'
    
    if ximg.sum() > 0:
        try:
            preprocessed = preprocess_brain_image(
                    ximg,
                    brain_extraction_modality='t1',
                    template_transform_type='TRSAA',
                    template=mni_T1_ANTSPY,
                    do_bias_correction=True,
                    do_denoising=False,
                    intensity_normalization_type='01',
                    verbose=False
                )
            preprocessed_image = preprocessed['preprocessed_image']
            preprocessed_brain_mask = preprocessed['brain_mask']
            preprocessed_brain_extracted = preprocessed_image * preprocessed_brain_mask
            preprocessed_brain_extracted.to_file(x_brain_extracted_path)
            print(f'{i} Done!')
        except Exception as ex:
            print(str(ex))
            print(i,path)
    else:
        print(i,path)


# In[ ]:


preprocessed_image = preprocessed['preprocessed_image']

preprocessed_brain_extracted = preprocessed['preprocessed_image'] * preprocessed['brain_mask']
# ants.plot(ximg)
# print(transformed_img)
ants.plot(preprocessed_image, axis=1)
ants.plot(preprocessed_brain_extracted, axis=1)


# In[ ]:


direction = preprocessed['preprocessed_image'].direction
direction


# ## Data Augmentation

# In[21]:


# paths
registered_paths = sorted(glob(f'{base_data_path}/*/nifty/*registered*.nii.gz'))
registered_brain_extracted_no_edges_paths = sorted(glob(f'{base_data_path}/*/nifty/**brain_extracted**.nii.gz'))
registered_brain_extracted_edges_paths = sorted(glob(f'{base_data_path}/*/nifty/*[!contrast]*brain_extracted*edges*.nii.gz'))
registered_denoised_brain_extracted_paths = sorted(glob(f'{base_data_path}/*/nifty/*denoised_brain_extracted*.nii.gz'))
print(len(registered_paths))
print(len(registered_brain_extracted_no_edges_paths))
print(len(registered_brain_extracted_edges_paths))
print(len(registered_denoised_brain_extracted_paths))


# In[8]:


seed = 1337
use_edges = True


# In[22]:


def get_data_paths(use_edges):
    icd_data_paths = sorted(glob(f'{base_data_path}/ICD/nifty/*brain_extracted*edges.nii.gz'))
    icd_data_paths, _ = train_test_split(icd_data_paths, test_size=0.25,random_state=seed)

    apathy_icd_data_paths = sorted(glob(f'{base_data_path}/apathy+ICD/nifty/*brain_extracted*edges.nii.gz'))
    apathy_data_paths = sorted(glob(f'{base_data_path}/apathy/nifty/*brain_extracted*edges.nii.gz'))

    pd_only_data_paths = sorted(glob(f'{base_data_path}/PD_Only/nifty/*brain_extracted*edges.nii.gz'))
    pd_only_data_paths, _ = train_test_split(pd_only_data_paths, test_size=0.82,random_state=seed)


    data_paths = icd_data_paths + apathy_icd_data_paths + apathy_data_paths + pd_only_data_paths
    data_paths = [path for path in data_paths if 'contrast' not in path]
    
    test_paths = sorted(glob(f'{base_data_path}/*/nifty/*registered_brain_extracted_edges.nii.gz'))
    test_paths = [path for path in test_paths if 'contrast' not in path]

    if not use_edges:
        test_paths = [f"{path[:-(len('_edges.nii.gz'))]}.nii.gz" for path in test_paths]
        data_paths = [f"{path[:-(len('_edges.nii.gz'))]}.nii.gz" for path in data_paths]

    data_paths = list(set(data_paths) - set(test_paths))
    
    print(f'Edges: {use_edges}')
    print(f'ICD: {len(icd_data_paths)}')
    print(f'Apathy+ICD: {len(apathy_icd_data_paths)}')
    print(f'Apathy only: {len(apathy_data_paths)}')
    print(f'PD only: {len(pd_only_data_paths)}')
    print(f'TRAIN/VAL paths: {len(data_paths)}')
    print(f'TEST paths: {len(test_paths)}')
    
    return data_paths, test_paths


# In[24]:


edges_paths, edges_test_paths = get_data_paths(use_edges=True)
print()
no_edges_paths, no_edges_test_paths = get_data_paths(use_edges=False)

# for source in edges_rotated:
#     match = '_edges_rotated_'
#     idx = source.index(match)
#     dest = f'{source[:idx]}_rotated_{source[idx+len(match):-7]}_edges.nii.gz'
    
#     os.rename(source, dest)

# edges_paths[:50]
# [x for x in no_edges_paths if '(1)' in x]
# sample_no_edges = '/home/carlos.jimenez/ai-in-health/ICD-Transformers/data/ICD/nifty/PPMI_3227_MR_AX_T1__br_raw_20130514095928238_50_S189273_I372195_registered_brain_extracted_rotated_(-1)_x.nii.gz'
# sample_edges = f'{sample_no_edges[:-7]}_edges.nii.gz'
print(edges_paths[0])
print(no_edges_paths[0])
# mni_sitk = sitk.ReadImage(edges_test_paths[0], sitk.sitkFloat32)
# # print(mni_sitk)
# plot_3d_overview(sitk.GetArrayFromImage(mni_sitk))

# mni_sitk = sitk.ReadImage(no_edges_test_paths[0], sitk.sitkFloat32)
# # print(mni_sitk)
# plot_3d_overview(sitk.GetArrayFromImage(mni_sitk))
edges_paths


# In[28]:


# registered_brain_extracted_no_edges_paths[:50]


# In[29]:


# registered_brain_extracted_edges_paths[:50]


# ## Rotations

# In[12]:


import random

from scipy import ndimage

@tf.function
def rotate(volume, angle, axes, mode):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        volume = ndimage.rotate(volume, angle, axes=axes, mode=mode, reshape=False)
        
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    
    return tf.numpy_function(scipy_rotate, [volume], tf.float32)


# In[13]:


def plot_rotation(plane, rotation_axis, angle, keep_direction=True):
    if plane == 'sagital':
        axis = 0
    elif plane == 'coronal':
        axis = 1
    elif plane == 'axial':
        axis = 2
    
    if rotation_axis == 'x':
        axes = (1,2)
    elif rotation_axis == 'y':
        axes = (0,2)
    elif rotation_axis == 'z':
        axes = (0,1)
    
#     figure, ax = plt.subplots(1, 2)

    test = ants.image_read(registered_paths[1])
    
    volume = rotate(test.numpy(), angle, axes, mode='nearest').numpy()
    if keep_direction:
        rotated_img = ants.from_numpy(volume, direction=test.direction)
    else:
        rotated_img = ants.from_numpy(volume)
    ants.plot(test, axis=axis)
    ants.plot(rotated_img, axis=axis)
    


# In[14]:


angles = [-15, -10, -5,-1, 1, 5, 10, 15]
# angles = [-1, 1]
plane = 'sagital'
rotation_axes = ['x','y']
# angle = 0
# plot_rotation(plane=plane, rotation_axis=rotation_axis, angle=angle, keep_direction=True)


# In[15]:


from itertools import combinations, chain
combs = [(x,y) for x in rotation_axes for y in angles]

def rotation_data_aumentation(paths):
    for i,path in enumerate(paths):
        if 'contrast' in path:
            continue
        ximg = ants.image_read(path)
        file_name = path[:-7]

        for axis,angle in combs:
            new_path = f'{file_name}_rotated_({angle})_{axis}.nii.gz'

            if axis == 'x':
                axes = (1,2)
            elif axis == 'y':
                axes = (0,2)
            elif axis == 'z':
                axes = (0,1)

            volume = rotate(ximg.numpy(), angle, axes, mode='nearest').numpy()
            rotated_img = ants.from_numpy(volume, direction=mni_T1_ANTSPY.direction)
            
            rotated_img.to_file(new_path)
            
        print(f'{i} Done!')


# In[17]:


# pd_only_xpaths[:10]


# In[ ]:


rotation_data_aumentation(pd_only_xpaths)


# In[137]:


# rotation_data_aumentation(registered_brain_extracted_edges_paths)


# In[ ]:


# rotation_data_aumentation(registered_denoised_brain_extracted_paths)


# ## Denoising

# In[32]:


# registered_paths = sorted(glob(f'{base_data_path}/P/nifty/*_registered.nii.gz'))
# print(len(registered_paths))
counter = 0
for path in pd_only_xpaths:
    if not 'brain_extracted' in path:
        ximg = ants.image_read(path)
        file_name = path[:-7]
        x_brain_extracted_denoised_path = file_name + '_registered_brain_extracted_denoised.nii.gz'
        try:
            preprocessed = preprocess_brain_image(
                ximg,
                brain_extraction_modality='t1',
                template_transform_type='TRSAA',
                template=mni_T1_ANTSPY,
                do_bias_correction=True,
                do_denoising=False,
                intensity_normalization_type='01',
                verbose=False
            )
#             preprocessed = preprocess_brain_image(
#                 ximg,
#                 truncate_intensity=None,
#                 brain_extraction_modality='t1',
#                 template_transform_type='TRSAA',
#                 template=mni_T1_ANTSPY,
#                 do_bias_correction=True,
#                 do_denoising=False,
#                 intensity_normalization_type='01',
#                 verbose=False
#             )
            
            preprocessed_image = preprocessed['preprocessed_image']
            preprocessed_brain_mask = preprocessed['brain_mask']
#             preprocessed_brain_extracted = preprocessed_image * preprocessed_brain_mask
#             preprocessed_brain_extracted.to_file(x_brain_extracted_contrast_path)
            
            # data augmentation 
            preprocessed_denoised_image = ants.denoise_image(preprocessed_image, preprocessed_brain_mask, shrink_factor=1)
            preprocessed_denoised_brain_extracted = preprocessed_denoised_image * preprocessed_brain_mask
#             print(x_brain_extracted_denoised_path)
            preprocessed_denoised_brain_extracted.to_file(x_brain_extracted_denoised_path)
            
            print(f'{i} done!')
            counter += 1
        except Exception as ex:
            print(str(ex))
            print(path)


# In[ ]:


ids = ['4070','3282', '3290', '4073', '3253','4072']
for id in ids:
    for path in xpaths:
        if f'_{id}_' in path:
            ximg = ants.image_read(path)
            file_name = path[:-7]
            x_registered_path = file_name + '_registered.nii.gz'
            x_brain_extracted_path = file_name + '_registered_brain_extracted.nii.gz'
            x_registered_denoised_path = file_name + '_registered_denoised.nii.gz'
            x_brain_extracted_denoised_path = file_name + '_registered_denoised_brain_extracted.nii.gz'

            if ximg.sum() > 0:
                try:
                    preprocessed = preprocess_brain_image(
                            ximg,
                            brain_extraction_modality='t1',
                            template_transform_type='TRSAA',
                            template=mni_T1_ANTSPY,
                            do_bias_correction=True,
                            do_denoising=False,
                            intensity_normalization_type='01',
                            verbose=False
                        )

                    preprocessed_image = preprocessed['preprocessed_image']
                    preprocessed_brain_mask = preprocessed['brain_mask']
                    preprocessed_brain_extracted = preprocessed_image * preprocessed_brain_mask

                    preprocessed_image.to_file(x_registered_path)
                    preprocessed_brain_extracted.to_file(x_brain_extracted_path)

                    # data augmentation 
                    preprocessed_denoised_image = ants.denoise_image(preprocessed_image, preprocessed_brain_mask, shrink_factor=1)
                    preprocessed_denoised_image.to_file(x_registered_denoised_path)
                    preprocessed_denoised_brain_extracted = preprocessed_denoised_image * preprocessed_brain_mask
                    preprocessed_denoised_brain_extracted.to_file(x_brain_extracted_denoised_path)
                    print(f'{i} done!')
                except Exception as ex:
                    print(str(ex))
                    print(path)
            else:
                print(path)
    


# Edge Detection


def edge_detection_data_augmentation(path):
    ximg = sitk.ReadImage(path, sitk.sitkFloat32)
    edges = sitk.SobelEdgeDetection(ximg)
    
    return edges


paths = sorted(glob(f'{base_data_path}/PD_Only/nifty/*brain_extracted*denoised*.nii.gz'))
print(len(paths))

for i, path in enumerate(paths):
#     print(path)
    file_name = path[:-7]
    brain_extracted_edges_path = f'{file_name}_edges.nii.gz'
    try:
#         print(brain_extracted_edges_path)
        edges = edge_detection_data_augmentation(path)
#         sitk.WriteImage(edges, brain_extracted_edges_path)
        print(f'{i} Done!')
    except Exception as ex:
        print(str(ex))
        print(i,path)


# In[239]:


minus1_rotated_paths = sorted(glob(f'{base_data_path}/*/nifty/*brain_extracted*(-1)*.nii.gz'))
one_rotated_paths = sorted(glob(f'{base_data_path}/*/nifty/*brain_extracted*(1)*.nii.gz'))
print(f'Total brain extracted: {len(minus1_rotated_paths)}')
print(f'Total brain extracted: {len(one_rotated_paths)}')
minus1_rotated_paths = [path for path in minus1_rotated_paths if 'edges' not in path]
one_rotated_paths = [path for path in one_rotated_paths if 'edges' not in path]
print(f'Total brain extracted: {len(minus1_rotated_paths)}')
print(f'Total brain extracted: {len(one_rotated_paths)}')

# minus1_rotated_paths

# rotated_paths

# edges_paths = sorted(glob(f'{base_data_path}/*/nifty/*edges*.nii.gz'))
# print(f'Total edges: {len(edges_paths)}')

# for i, path in enumerate(brain_extracted_paths):
#      if 'denoise' in path:
#             counter += 1
# print(f'Total denoised: {counter}')


# for i, path in enumerate(rotated_paths):
#     file_name = path[:-(len(".nii.gz"))]
#     brain_extracted_edges_detected_path = f'{file_name}_edges.nii.gz'
# #     print(brain_extracted_edges_detected_path)
#     edges = edge_detection_data_augmentation(path)
# #     plot_3d_overview(sitk.GetArrayFromImage(edges))
#     sitk.WriteImage(edges, brain_extracted_edges_detected_path)
#     print(f'{i+1}/{len(rotated_paths)} done!')


# ## Flipping

# In[ ]:


test_img = ants.image_read(registered_paths[1])
print(test_img.direction)

def flipping_data_augmentation(path):
    ximg = sitk.ReadImage(path, sitk.sitkFloat32)
    flipped_img = sitk.Flip(ximg,(True, False, False))
    
    return flipped_img


# In[ ]:


brain_extracted_edges_paths = sorted(glob(f'{base_data_path}/ICD/nifty/*brain_extracted.nii.gz'))
brain_extracted_edges_paths[:28]
    
for path in brain_extracted_edges_paths[:1]:
    test_img = ants.image_read(path)
    ants.plot(test_img, axis=0)
    flipped = flipping_data_augmentation(path)
    sitk.WriteImage(flipped, '/home/carlos.jimenez/flipped.nii.gz')
    test2 = ants.image_read('/home/carlos.jimenez/flipped.nii.gz')
    ants.plot(test2, axis=0)


# ## The whole thing

# In[ ]:


# x3d = sitk.HistogramMatching(ximg, mni152_T1) # Histogram Matching
# x3d = sitk.Multiply(x3d, mni152_brain_mask) # mask brain (Extraction)
# x3d = sitk.CurvatureAnisotropicDiffusion(x3d, conductanceParameter=1, numberOfIterations=1) # denoise a bit
# # x3d = sitk.SobelEdgeDetection(x3d)
# plot_3d_overview(sitk.GetArrayFromImage(ximg))
# plot_3d_overview(sitk.GetArrayFromImage(x3d))


# ## Analyze directions

# In[ ]:


for i, xpath in enumerate(pd_only_xpaths):
  ants_image = ants.image_read(xpath)
  if not np.array_equal(ants_image.direction.flatten(), np.array([-1, 0, 0, 0, -1, 0, 0, 0, 1])):
    print(i,xpath)
    print(np.array([-1, 0, 0, 0, -1, 0, 0, 0, 1]), ants_image.direction.flatten())


# In[ ]:





# In[ ]:


get_ipython().system(" find data -iname '*contrast*'")


# In[ ]:



preprocessed_image = ants.apply_transforms(fixed = template_image, moving = preprocessed_image,
                transformlist=registration['fwdtransforms'], interpolator="linear", verbose=verbose)
            mask = ants.apply_transforms(fixed = template_image, moving = mask,
                transformlist=registration['fwdtransforms'], interpolator="genericLabel", verbose=verbose)    

