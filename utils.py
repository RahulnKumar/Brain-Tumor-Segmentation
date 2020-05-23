# This preprocessing part of this project in not written by me
# Original author is : https://github.com/polo8214/Brain-tumor-segmentation-using-deep-learning


# Imports 
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import SimpleITK as sitk
import skimage.io as io
import skimage.transform as trans

# Initializations
smooth = 1 
num_of_aug = 1
num_epoch = 10
img_size = 120

# Applying bias correction

def n4itk(img):         #must input with sitk img object
    img = sitk.Cast(img, sitk.sitkFloat32)
    img_mask = sitk.BinaryNot(sitk.BinaryThreshold(img, 0, 0))   ## Create a mask spanning the part containing the brain, as we want to apply the filter to the brain image
    corrected_img = sitk.N4BiasFieldCorrection(img, img_mask)
    return corrected_img 


# Loading data function

import glob
def create_data(src, mask, label=False, resize=(155,img_size,img_size)):
    files = glob.glob(src + mask, recursive=True)
    imgs = []
    print('Processing---', mask)
    for file in files:
        img = io.imread(file, plugin='simpleitk')
        img = trans.resize(img, resize, mode='constant')
        if label:
           
            #img[img != 1] = 0       #Enhancing tumor => set 4 = 1 and every thing else to 0
            #img[img == 4] = 1       
            #img[img == 2] = 0       #Core tumor => 1+3+4 is set to 1 and every thing else to 0
            #img[img !=0] = 1
            img[img != 0] = 1       #Complete tumor =>1+2+3+4 is set to 1 and o is just normal tissue
            img = img.astype('float32')
        else:
            img = (img-img.mean()) / img.std()      #normalization => zero mean   !!!care for the std=0 problem
        for slice in range(50,130):
            img_t = img[slice,:,:]
            img_t =img_t.reshape((1,)+img_t.shape)
            img_t =img_t.reshape((1,)+img_t.shape)   #become rank 4
            img_g = augmentation(img_t,num_of_aug)
            for n in range(img_g.shape[0]):
                imgs.append(img_g[n,:,:,:])
    name = 'y_'+ str(img_size) if label else 'x_'+ str(img_size)
    np.save(name, np.array(imgs).astype('float32'))  # save at home
    print('Saved', len(files), 'to', name)
    
# Data augmentation

def augmentation(scans,n):          #input img must be rank 4 
    datagen = ImageDataGenerator(
        featurewise_center=False,   
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=25,   
        #width_shift_range=0.3,  
        #height_shift_range=0.3,   
        horizontal_flip=True,   
        vertical_flip=True,  
        zoom_range=False)
    i=0
    scans_g=scans.copy()
    for batch in datagen.flow(scans, batch_size=1, seed=1000): 
        scans_g=np.vstack([scans_g,batch])
        i += 1
        if i == n:
            break
    '''    remember arg + labels  
    i=0
    labels_g=labels.copy()
    for batch in datagen.flow(labels, batch_size=1, seed=1000): 
        labels_g=np.vstack([labels_g,batch])
        i += 1
        if i > n:
            break    
    return ((scans_g,labels_g))'''
    return scans_g
#scans_g,labels_g = augmentation(img,img1, 10)
#X_train = X_train.reshape(X_train.shape[0], 1, img_size, img_size)