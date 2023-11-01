# Introduction
&nbsp;&nbsp;&nbsp; DICOM stands for digital image and communication in medicine and is the standard way of storing medical data, which consists of the original array of images, as well as meta-data, which stores information regarding pixel spacing, slice thickness, affine, acquisition time, etc. 

&nbsp;&nbsp;&nbsp; The images are taking using a computed tomography (CT) scan. X-ray beams are released and travels through the human body to the detector on the other side. Denser parts of the body such as the bones absorb more radiation than other parts such as soft tissues and air. The rays that are absorbed do not reach the detector as well, causing denser parts of the body to show up in lighter shades of gray in the image scans. The degree of the x-ray absorption is measured in Hounsfield Units (HU). The HU for general parts of the body are rouhgly -1000HU for air, -500HU for lungs, -200HU for fat, 0HU for water, 50HU for soft tissue, and >500HU for bone. 

&nbsp;&nbsp;&nbsp; This repository goes through several different image analysis, such as reading and handling medical image data sets and image segmentations using different techniques.

---
## Table of Contents

---
## DICOM Image Analysis
* [Code for LUNG/CHEST IMAGES](https://github.com/jlee92603/medical_image_exploration/blob/main/chest%20lung%20images.ipynb)
* [Code for PELVIS IMAGES](https://github.com/jlee92603/medical_image_exploration/blob/main/pelvis%20images.ipynb)
* [Code for ANKLE IMAGES](https://github.com/jlee92603/medical_image_exploration/blob/main/ankle%20images.ipynb)

&nbsp;&nbsp;&nbsp; Several functions were performed on the DICOM format medical images to explore and model the data. Notable methods include rescaling image based on pixels to Hounsfield Units, creating a histogram based on HU, display of a sample stack of the images, resampling the images to be isovoxel, creating a 3D visual model of the slices, and making a mask around the lungs. 

&nbsp;&nbsp;&nbsp; In addition to the functions in the examples above, there is an additional function that allows 3D interactive visual model of the images. The resulting interactive model's file size was too big to include; hence, a screen recording of the interactive model is attached below. 

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
Slice thickness: 2.500000

Pixel Spacing (row, col): (0.703125, 0.703125)

Shape before resampling (126, 512, 512)

Shape after resampling (315, 360, 360)


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


### Function for an interactive 3D visual model using plotly
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
### Function for interactive 3D visual model using plotly
In addition to the functions above, an interactive 3D visual model is created using plotly. The code is shown below, as well as the demonstration of the interactive visual model. 
<img width="710" alt="Screen Shot 2023-06-01 at 2 25 28 PM" src="https://github.com/jlee92603/DICOM_EDA/assets/70551445/336d285a-994c-4005-a0d4-effb89ea2378">

#### The demonstration of the interactive 3D visual model (lung/chest)
https://github.com/jlee92603/DICOM_EDA/assets/70551445/41c0b85f-3422-4649-8213-b8e9e1f3e8c9

#### The demonstration of the interactive 3D visual model (pelvis)
https://github.com/jlee92603/DICOM_EDA/assets/70551445/5e292990-62a6-488c-aee1-5884a3b855ba

#### The demonstration of the interactive 3D visual model (ankle)
https://github.com/jlee92603/DICOM_EDA/assets/70551445/464b9441-bcee-4bc4-990b-f803844c12e9

---
## Segmentation of Lung Images
* [Code for LUNG SEGMENTATIONS](https://github.com/jlee92603/medical_image_exploration/blob/main/segmentation%20of%20lungs.ipynb)

&nbsp;&nbsp;&nbsp; Lung and vessel segmentations done on NIFTI format medical images by creating contours. Notable methods include finding contour of lungs and vessels, determining area of lungs and vessels, creating a mask from the contours and overlaying mask ontop of original image, and ploting both original image with contour and lung mask.

---
## Applying watershed algorithm and active contour 
* [Code for WATERSHED and ACTIVE CONTOUR](https://github.com/jlee92603/medical_image_exploration/blob/main/watershed%20and%20active%20contour.ipynb)

&nbsp;&nbsp;&nbsp; Watershed is an image processing transformation on a grayscale image that helps with segmentation or differentiating different objects in an image. This algorithm treats the image as a topographic map, where how dark or light a point on the image is represents the height on a topographic map. 

&nbsp;&nbsp;&nbsp; Active contour models are used to trace out the outline of an object from an image. This model is widely used in tracking objects, recognizing shapes, segmentation, and detecting edges. 

---
