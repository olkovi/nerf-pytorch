#!/usr/bin/env python
# coding: utf-8

# In[5]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import math
import os
import time
print('Do you know the camera parameters? If not, change them here!')
def get_intr(fx, fy, sz_x, sz_y, npix_x, npix_y):
    """
    f, sz_x, sz_y, npix_x, npix_y == focal length, sensor_size_x, sensor_size_y, npix_x, npix_y
    """
    to_pix = 1/(sz_x/npix_x) #m_to pix
    fx_pix = fx*to_pix#focal length in pixels
    fy_pix = fy*to_pix
    intr = np.array([[fx_pix, 0.000000e+00, npix_x/2, 0.000000e+00],
                     [0.000000e+00, fy_pix, npix_y/2, 0.000000e+00],
                    [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00],
                     [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]])

    intr_inv = np.linalg.inv(intr)
    return intr, intr_inv

def sharpness(imagePath):
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	return fm

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()


# In[23]:

full_format = input('Full Format?'+'\nY/n ') == 'Y' #full nerf transforms format or shorthand
fx, fy, sz_x, sz_y, npix_x, npix_y =0.002, 0.002, 3.896*10**-3, 2.453*10**-3, 1920, 1080

to_pix = 1/(sz_x/npix_x) #to_pixel conversion factor

camera_angle_x = 2*math.atan(0.5*sz_x/fx) #x fov
camera_angle_y = 2*math.atan(0.5*sz_y/fy) #x fov

fl_x = fx*to_pix
fl_y = fy*to_pix

k1, k2 = 0, 0
p1, p2 = 0, 0
cx, cy = npix_x/2, npix_y/2

w = npix_x
h = npix_y

intr, intr_inv = get_intr(fx, fy, sz_x, sz_y, npix_x, npix_y) 


# In[117]:
#loc = input('Images in "train", "test" or "image"?')
loclist = ['train', 'test', 'image']
locs = []

dir_list = next(os.walk('.'))[1]
print(dir_list)
#check if train, test or image directories present, if so add dir to list to be processed
for i in range(len(dir_list)):
    isinlist=0
    for j in loclist:
        if j in next(os.walk(os.path.join('.',dir_list[i])))[1]:
            locs.append(j)
            break
        else:
            isinlist+=1
        if isinlist==3:
            dir_list[i] = 0
dir_new = [i for i in dir_list if i != 0] 

print(dir_new)
# In[119]:
print(len(dir_new))
print(len(locs))
assert len(dir_new) == len(locs), "Lengths Unequal!"


for i in range(len(dir_new)):
    source_dir = dir_new[i]
    loc = locs[i]
    print(loc)
    image_dir = loc
    image_dir = os.path.join(source_dir, image_dir)
    a = np.load(os.path.join(source_dir,'cameras_sphere.npz'))
    file_names = os.listdir(image_dir)
    assert int(len(a)/4) == len(file_names), "Number of Images does not match the number of Poses"
    aabb_scale = a['scale_mat_0'][0][0]



    import time
    diftime = time.time()
    sharp_list = []
    print('Getting sharpness')
    for file_name in file_names:
        sharp_list.append(sharpness(os.path.join(image_dir, file_name)))
        print(file_name)
    print(time.time()-diftime)


    if full_format:
        out_data = {
            'camera_angle_x': camera_angle_x,
            'camera_angle_y': camera_angle_y,
            'fl_x': fl_x,
            'fl_y': fl_y,
            'k1': k1,
            'k2': k2,
            'p1': p1,
            'p2': p2,
            'cx': cx,
            'cy': cy,
            'w': w,
            'h': h,
            'aabb_scale': aabb_scale
            }
    else:
        out_data = {
            'camera_angle_x': camera_angle_x
        }
    out_data['frames'] = []
    np.set_printoptions(suppress = True)
    import json
    for i in range(len(file_names)):
        extmat_blender = a['world_mat_inv_{}'.format(i)]@intr #Neus Extrinsic matrix (can only be acquired if intrinsic known)
        # extmat_blender = intr_inv@a['world_mat_{}'.format(i)] #Neus Extrinsic matrix (can only be acquired if intrinsic known)
        #extmat_blender[1], extmat_blender[0] = extmat_blender[0], extmat_blender[1]
        #extmat_blender[1] = -extmat_blender[1]
        extmat_blender[:3,1] = -extmat_blender[:3,1]
        extmat_blender[:3,2] = -extmat_blender[:3,2]
        #extmat_blender[3,1] = -extmat_blender[3,1]
        #extmat_blender[3,2] = -extmat_blender[3,2]
        print(str(i)+'\n',extmat_blender)
        #extmat_blender[3] = -extmat_blender[3] # blender inverts y and z axes compared to neus, invert these to get blender Extrinsic matrix
        frame_data = {
            'file_path': "./"+os.path.join(loc,file_names[i]).replace("\\","/").replace(".png",""),
            'sharpness': sharp_list[i],
            'transform_matrix': np.ndarray.tolist(extmat_blender)
        }
        out_data['frames'].append(frame_data)

    
    with open(source_dir + '/' + 'transforms_'+ loc + '.json' if loc != 'image' else source_dir + '/' + 'transforms.json', 'w') as out_file:
            json.dump(out_data, out_file, indent=4)

print('Do you know the camera parameters? If not, change them here!')
# In[ ]:


# In[3]:




