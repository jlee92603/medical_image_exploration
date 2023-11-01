---
# Medical Image Exploration
## Introduction
This project project goes through several different image analysis techniques, such as reading and handling medical image data sets and image segmentation using different techniques. 

DICOM stands for digital image and communication in medicine and is the standard way of storing medical data, which consists of the original array of images, as well as meta-data, which stores information regarding pixel spacing, slice thickness, affine, acquisition time, etc. DICOM images are used in this project. 

The images are taking using a computed tomography (CT) scan. X-ray beams are released and travels through the human body to the detector on the other side. Denser parts of the body such as the bones absorb more radiation than other parts such as soft tissues and air. The rays that are absorbed do not reach the detector as well, causing denser parts of the body to show up in lighter shades of gray in the image scans. The degree of the x-ray absorption is measured in Hounsfield Units (HU). The HU for general parts of the body are rouhgly -1000HU for air, -500HU for lungs, -200HU for fat, 0HU for water, 50HU for soft tissue, and >500HU for bone. 

---
## Table of Contents
- [DICOM Image Analysis](#DICOM-Image-Analysis)
    - [Import Necessary Packages](#Import-Necessary-Packages)
    - [Load Images](#Load-Images)
    - [Relevant Functions](#Relevant-Functions)
    - [HU Histogram](#HU-Histogram)
    - [Display Images](#Display-Images)
    - [Resample the Images](#Resample-the-Images)
    - [Segmentation using a mask](#Segmentation-using-a-mask)
    - [3D Visual Plotting](#3D-Visual-Plotting)
    - [Interactive 3D Visual Model](#Interactive-3D-Visual-Model)
        - [Visual Model for Chest](#Visual-Model-for-Chest)
        - [Visual Model for Pelvis](#Visual-Model-for-Pelvis)
        - [Visual Model for Ankle](#Visual-Model-for-Ankle)
- [Segmentation of Lung Images](#Segmentation-of-Lung-Images)
    - [Import Necessary Packages](#Import-Necessary-Packages-for-Segmentation)
    - [Load Images](#Load-Lung-Images)
    - [Relevant Functions](#Relevant-Segmentation-Functions)
    - [Display Images](#Display-Lung-Images)
    - [Lung Segmentation](#Lung-Segmentation)
    - [Vessel Segmentation](#Vessel-Segmentation)
- [Watershed and Active Contour](#Applying-watershed-algorithm-and-active-contour)

---
## DICOM Image Analysis
* [Code for LUNG/CHEST IMAGES](https://github.com/jlee92603/medical_image_exploration/blob/main/chest%20lung%20images.ipynb)
* [Code for PELVIS IMAGES](https://github.com/jlee92603/medical_image_exploration/blob/main/pelvis%20images.ipynb)
* [Code for ANKLE IMAGES](https://github.com/jlee92603/medical_image_exploration/blob/main/ankle%20images.ipynb)

Several functions were performed on the DICOM format medical images to explore and model the data. Notable methods include rescaling image based on pixels to Hounsfield Units, creating a histogram based on HU, display of a sample stack of the images, resampling the images to be isovoxel, creating a 3D visual model of the slices, and making a mask around the lungs. 

In addition to the functions in the examples above, there is an additional function that allows 3D interactive visual model of the images. The resulting interactive model's file size was too big to include; hence, a screen recording of the interactive model is attached below. 

The code for the lung/chest images is explained in more detail below. 
### Import Necessary Packages
```
%matplotlib inline
import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import figure_factory as FF
from plotly.graph_objs import *
init_notebook_mode(connected=True) 
```
### Load Images
The image files are from The Cancer Imaging Archive (TCIA) from the NIH National Cancer Institute. 
```
data_path = "[insert path to image file data here]"
output_path = working_path = "[insert working path here]"
g = glob(data_path + '/*.dcm')
```
### Relevant Functions
```
# loop over image files and store everything into a list
def load_scan(path):    
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)] # os.listdir() gets list of all files and directories
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        # ImagePositionPatient returns x,y,z coor of upper left hand corner
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        # SliceLocation returns relative position of image plance in mm
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        # unit of distance in z-dir per slice (image)
        # less than 2mm; in order to optimize as much info
        s.SliceThickness = slice_thickness
        
    return slices

# convert pixels to Hounsfield Units
def get_pixels_hu(scans): # input is output from load_scans
    
    # pixel_array 
    # stack: join sequence of arrays along a new axis
    image = np.stack([s.pixel_array for s in scans])
    
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    # b in relationship between stored values SV and output units (m*SV + b)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

id=0
patient = load_scan(data_path)
imgs = get_pixels_hu(patient)

np.save(output_path + "/" + "fullimages_%d.npy" % (id), imgs)
```
### HU Histogram 
The histogram for the HU from the medical images is plotted below. HU is the scale for describing radiodensity for medical images. 
```
# displaying histogram based on HU
file_used=output_path+"/"+"fullimages_%d.npy" % id
imgs_to_process = np.load(file_used).astype(np.float64) 

# flatten flattens the array: [[1,2], [3,4]] -> [1,2,3,4]
plt.hist(imgs_to_process.flatten(), bins=50, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()
```
This histogram shows lots of air, some lungs, lots of soft tissue (mostly muscle, liver, some fat), and some bone. 
<img width="562" alt="Screen Shot 2023-10-31 at 11 28 39 PM" src="https://github.com/jlee92603/medical_image_exploration/assets/70551445/c952a863-73d3-45b6-80aa-f31ff6b6873a">

### Display Images
A couple images are selected and displayed in a stack. 
```
# displaying image stack
id = 0
imgs_to_process = np.flip(np.load(output_path + '/' + 'fullimages_{}.npy'.format(id)))

def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=3):
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        # imshow: display data as image
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()

sample_stack(imgs_to_process)
```
<img width="553" alt="Screen Shot 2023-10-31 at 11 29 32 PM" src="https://github.com/jlee92603/medical_image_exploration/assets/70551445/0989bee3-1841-4f83-a842-cacbfe22b399">

### Resample the Images
The images are resampled to be isovoxel, or have roughly the same number of voxels in each dimension. 
```
# resampling
print("Slice Thickness: %f" % patient[0].SliceThickness)
# PixelSpacing returns the physical distance in mm between center of each pixel
# first value is space between rows (vertical distance); second value is space between col (horizontal distance)
print("Pixel Spacing (row, col): (%f, %f) " % (patient[0].PixelSpacing[0], patient[0].PixelSpacing[1]))

id = 0
imgs_to_process = np.flip(np.load(output_path+'/' + 'fullimages_{}.npy'.format(id)))

# diff voxel resolution between patients; resample it to make it isovoxel
def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    # change each item in list to float; create numpy array from list
    # spacing = list of distance between pixels in z, x, y directions
    spacing = map(float, ([scan[0].SliceThickness] + list(scan[0].PixelSpacing)))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing # array 
    new_real_shape = image.shape * resize_factor #shape: returns dimensions of image
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.zoom(image, real_resize_factor)
    
    return image, new_spacing

print("Shape before resampling\t", imgs_to_process.shape)
imgs_after_resamp, spacing = resample(imgs_to_process, patient, [1,1,1])
print("Shape after resampling\t", imgs_after_resamp.shape)
```
```
Slice thickness: 2.500000
Pixel Spacing (row, col): (0.703125, 0.703125)
Shape before resampling (126, 512, 512)
Shape after resampling (315, 360, 360)
```

The slices are 2.5mm thick and each voxel represents 0.7mm. 
The CT is reconstructed at 512x512 voxels, which each slice representing approximately 370mm of data in length and width. 

### Segmentation using a mask
A mask is created around the lung region in the image to perform a segmentation of the lungs (or to separate the lungs). 
```
# segmentation
def make_lungmask(img, display=False):
    row_size= img.shape[0]
    col_size = img.shape[1]
    
    # Standardize the pixel values by subtracting the mean and dividing by the standard deviation
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std 
    
    # Find the average pixel value near the lungs to renormalize washed out images
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, move underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean

    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    # k means: method that aims to partition n observations into k clusters where each
    # observation belongs to the clusters w the nearest mean
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers) # threshold between soft tissue/bone and lung/air
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img,np.ones([3,3])) #erode away boundaries of regions of foreground pixels
    dilation = morphology.dilation(eroded,np.ones([8,8])) #enlarge boundaries of regions of foreground pixels

    labels = measure.label(dilation) # Different labels are displayed in different colors
    label_vals = np.unique(labels) # return unique elements in array
    regions = measure.regionprops(labels) # measure properties of labeled image regions
    good_labels = []
    for prop in regions:
        B = prop.bbox # bounding box
        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0

    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation

    if (display):
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask*img, cmap='gray')
        ax[2, 1].axis('off')
        
        plt.show()
    return mask*img

# single slice example at each step
img = imgs_after_resamp[260]
make_lungmask(img, display=True)

# apply make to all slices
masked_lung = []

for img in imgs_after_resamp:
    masked_lung.append(make_lungmask(img))

sample_stack(masked_lung, start_with=164, show_every=3)
```
<img width="533" alt="Screen Shot 2023-10-31 at 11 35 46 PM" src="https://github.com/jlee92603/medical_image_exploration/assets/70551445/31134518-8a49-43bf-a4a2-158f9a29d910">
<img width="604" alt="Screen Shot 2023-10-31 at 11 36 04 PM" src="https://github.com/jlee92603/medical_image_exploration/assets/70551445/4660a637-6916-4c9c-8175-90b00cbace29">


### 3D Visual Plotting
A mesh is made. The mesh is used to create a 3D visualization of the images, with each slice stacked on top of each other to dislay a 3 dimensional visual. 
```
# 3D plotting
def make_mesh(image, threshold=-300, step_size=1):

    print("Transposing surface") 
    # original axes is 0,1,2; tranposed so axis 0 takes place of axis 2 and vice versa
    p = image.transpose(2,1,0)
    
    print("Calculating surface")
    # marching cubtes algorithm is used to generate 3D mesh from dataset
    verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True) 
    return verts, faces

def plt_3d(verts, faces):
    print("Drawing")
    x,y,z = zip(*verts) 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    ax.set_facecolor((0.7, 0.7, 0.7))
    
    ax.invert_yaxis()
    
    plt.show()

v, f = make_mesh(imgs_after_resamp, 350)
plt_3d(v, f)
```
### Interactive 3D Visual Model
In addition to the functions above, an interactive 3D visual model is created using plotly. The code is shown below, as well as the demonstration of the interactive visual model. 
<img width="710" alt="Screen Shot 2023-06-01 at 2 25 28 PM" src="https://github.com/jlee92603/DICOM_EDA/assets/70551445/336d285a-994c-4005-a0d4-effb89ea2378">

#### Visual Model for Chest
The demonstration of the interactive 3D visual model for chest
https://github.com/jlee92603/DICOM_EDA/assets/70551445/41c0b85f-3422-4649-8213-b8e9e1f3e8c9

#### Visual Model for Pelvis
The demonstration of the interactive 3D visual model for pelvis
https://github.com/jlee92603/DICOM_EDA/assets/70551445/5e292990-62a6-488c-aee1-5884a3b855ba

#### Visual Model for Ankle
The demonstration of the interactive 3D visual model for ankle
https://github.com/jlee92603/DICOM_EDA/assets/70551445/464b9441-bcee-4bc4-990b-f803844c12e9

---
## Segmentation of Lung Images
* [Code for LUNG SEGMENTATIONS](https://github.com/jlee92603/medical_image_exploration/blob/main/segmentation%20of%20lungs.ipynb)

Lung and vessel segmentations done on NIFTI format medical images by creating contours. Notable methods include finding contour of lungs and vessels, determining area of lungs and vessels, creating a mask from the contours and overlaying mask ontop of original image, and ploting both original image with contour and lung mask.

### Import Necesary Packages for Segmentation
```
# read data
import os
import shutil

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from skimage import measure

import glob
import csv
```
### Load Lung Images
```
basepath = '/Users/jihye/Projects/research/Images/slice*.nii.gz'
paths = sorted(glob.glob(basepath))
print('Images found:', len(paths))
```
### Relevant Segmentation Functions
Please refer to the 'segmentation of lungs.ipynb' file for relevant functions. 

### Display Lung Images
```
# display sample slice
for c, exam_path in enumerate(paths):
    ct_img = nib.load(exam_path)
    ct_numpy = ct_img.get_fdata() # get array data
    
    if c == 1:
        # show the first slice
        fig,ax = plt.subplots(1,3)
        ax[0].set_title('original')
        ax[0].imshow(ct_numpy.T, cmap="gray", origin="lower")
        ax[0].axis('off')
        
        # show the first slice with emphasis on tissues (HU level 50, window 250)
        ax[1].set_title('+50,250')
        ax[1].imshow(ct_numpy.clip(-75,175).T, cmap="gray", origin="lower")
        ax[1].axis('off')
        
        # show the first slice with emphasis on lungs (HU level -600, window 1500)
        ax[2].set_title('-600,1500')
        ax[2].imshow(ct_numpy.clip(-1350,175).T, cmap="gray", origin="lower")
        ax[2].axis('off')
        plt.show()
        break
```
<img width="342" alt="Screen Shot 2023-10-31 at 11 43 06 PM" src="https://github.com/jlee92603/medical_image_exploration/assets/70551445/29634fe8-90e7-4cf8-b70b-decfbc634727">

### Lung Segmentation
```
# lung segmentation based on image intensity and medical image processing
outpath = './LUNGS/'
contour_path = './Contours/'
paths = sorted(glob.glob(basepath))
myFile = open('lung_volumes.csv', 'w')
lung_areas = []
make_dirs(outpath)
make_dirs(contour_path)
i = 0

fig,ax = plt.subplots(7,6,figsize=[12,12])

for c, exam_path in enumerate(paths):
    img_name = exam_path.split("/")[-1].split('.nii')[0]
    out_mask_name = outpath + img_name + "_mask"
    contour_name = contour_path + img_name + "_contour"

    # 1. find pixel dimensions to calculate the area in mm^2
    ct_img = nib.load(exam_path)
    pixdim = find_pix_dim(ct_img)
    ct_numpy = ct_img.get_fdata()

    # 2. binarize image using intensity thresholding 
    min_HU = -1000
    max_HU = -300
    
    # 3. contour finding (contour is set of points that describe line or area)
    contours = intensity_seg(ct_numpy, min_HU, max_HU) # expected HU range

    # 4. find lungs of set of possible contours and plot
    lungs = find_lungs(contours)
    ax[int(i/6),int(i%6)].imshow(ct_numpy.T, cmap=plt.cm.gray)
    ax[int(i/6),int(i%6)].axis('off')
    
    # show contours and plot
    for contour in lungs:
        ax[int(i/6),int(i%6)].plot(contour[:,0], contour[:,1], linewidth=1)
    i+=1
    
    # 5. contour to binary mask
    lung_mask = create_mask_from_polygon(ct_numpy, lungs)
    save_nifty(lung_mask, out_mask_name, ct_img.affine)
    
    # plot
    ax[int(i/6),int(i%6)].imshow(lung_mask.T, cmap="gray", origin="lower")
    ax[int(i/6),int(i%6)].axis('off')
    i+=1

    # compute lung area
    lung_area = compute_area(lung_mask, find_pix_dim(ct_img))
    lung_areas.append([img_name,lung_area]) # int is ok since the units are already mm^2
    print(img_name,'lung area in mm^2:', lung_area)

plt.show()
```
<img width="583" alt="Screen Shot 2023-10-31 at 11 45 22 PM" src="https://github.com/jlee92603/medical_image_exploration/assets/70551445/d0fa3f96-cc4a-4d3e-9c7b-736f5eaa0343">

### Vessel Segmentation
```
fig,ax = plt.subplots(7,6,figsize=[12,12])
i=0
    
for c, exam_path in enumerate(paths):
    img_name = exam_path.split("/")[-1].split('.nii')[0]
    vessel_name = vessels + img_name + "_vessel_only_mask"
    overlay_name = overlay_path + img_name + "_vessels"

    # 1. find pixel dimensions to calculate the area in mm^2
    ct_img = nib.load(exam_path)
    pixdim = find_pix_dim(ct_img)
    ct_numpy = ct_img.get_fdata()
    
    # 2. contour finding
    contours = intensity_seg(ct_numpy, -1000, -300)

    # 4. find lungs from set of possible contours and create a mask
    lungs_contour = find_lungs(contours)
    lung_mask = create_mask_from_polygon(ct_numpy, lungs_contour)

    # 5. compute lung area
    lung_area = compute_area(lung_mask, find_pix_dim(ct_img))

    # 6. create a mask of vessels
    vessels_only = create_vessel_mask(lung_mask, ct_numpy, denoise=False)
    ax[int(i/6),int(i%6)].imshow(vessels_only.T, 'gray', origin="lower")
    ax[int(i/6),int(i%6)].axis('off')
    i+=1
    
    # 7. plot image with vessel mask
    ax[int(i/6),int(i%6)].imshow(ct_numpy.T, 'gray', interpolation='none')
    ax[int(i/6),int(i%6)].imshow(np.flipud(vessels_only.T), 'jet', interpolation='none', alpha=0.5)
    ax[int(i/6),int(i%6)].axis('off')
    i+=1
        
    # 7. compute vessel area as well as ratio of vessel area to lung area
    vessel_area = compute_area(vessels_only, find_pix_dim(ct_img))
    ratio = (vessel_area / lung_area) * 100
    print(img_name, 'Vessel %:', ratio)
    lung_areas_csv.append([img_name, lung_area, vessel_area, ratio])
    ratios.append(ratio)
    
plt.show()
```
<img width="581" alt="Screen Shot 2023-10-31 at 11 45 36 PM" src="https://github.com/jlee92603/medical_image_exploration/assets/70551445/d878ca63-b9ba-4b4d-b4c3-03c7f71b6073">

---
## Applying watershed algorithm and active contour 
Watershed and active contour techniques are explored on several images. 
* [Code for WATERSHED and ACTIVE CONTOUR](https://github.com/jlee92603/medical_image_exploration/blob/main/watershed%20and%20active%20contour.ipynb)

Watershed is an image processing transformation on a grayscale image that helps with segmentation or differentiating different objects in an image. This algorithm treats the image as a topographic map, where how dark or light a point on the image is represents the height on a topographic map. 

Active contour models are used to trace out the outline of an object from an image. This model is widely used in tracking objects, recognizing shapes, segmentation, and detecting edges. 

---
