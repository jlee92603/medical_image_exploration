# Introduction
&nbsp;&nbsp;&nbsp; DICOM stands for digital image and communication in medicine and is the standard way of storing medical data, which consists of the original array of images, as well as meta-data, which stores information regarding pixel spacing, slice thickness, affine, acquisition time, etc. 

&nbsp;&nbsp;&nbsp; The images are taking using a computed tomography (CT) scan. X-ray beams are released and travels through the human body to the detector on the other side. Denser parts of the body such as the bones absorb more radiation than other parts such as soft tissues and air. The rays that are absorbed do not reach the detector as well, causing denser parts of the body to show up in lighter shades of gray in the image scans. The degree of the x-ray absorption is measured in Hounsfield Units (HU). The HU for general parts of the body are rouhgly -1000HU for air, -500HU for lungs, -200HU for fat, 0HU for water, 50HU for soft tissue, and >500HU for bone. 

---
## DICOM Image Analysis
* [Code for LUNG/CHEST IMAGES](https://github.com/jlee92603/medical_image_exploration/blob/main/chest%20lung%20images.ipynb)
* [Code for PELVIS IMAGES](https://github.com/jlee92603/medical_image_exploration/blob/main/pelvis%20images.ipynb)
* [Code for ANKLE IMAGES](https://github.com/jlee92603/medical_image_exploration/blob/main/ankle%20images.ipynb)

&nbsp;&nbsp;&nbsp; Several functions were performed on the DICOM format medical images to explore and model the data. Notable methods include rescaling image based on pixels to Hounsfield Units, creating a histogram based on HU, display of a sample stack of the images, resampling the images to be isovoxel, creating a 3D visual model of the slices, and making a mask around the lungs. 

&nbsp;&nbsp;&nbsp; In addition to the functions in the examples above, there is an additional function that allows 3D interactive visual model of the images. The resulting interactive model's file size was too big to include; hence, a screen recording of the interactive model is attached below. 

### Function for an interactive 3D visual model using plotly
<img width="710" alt="Screen Shot 2023-06-01 at 2 25 28 PM" src="https://github.com/jlee92603/DICOM_EDA/assets/70551445/336d285a-994c-4005-a0d4-effb89ea2378">

### The demonstration of the interactive 3D visual model (lung/chest)
https://github.com/jlee92603/DICOM_EDA/assets/70551445/41c0b85f-3422-4649-8213-b8e9e1f3e8c9

### The demonstration of the interactive 3D visual model (pelvis)
https://github.com/jlee92603/DICOM_EDA/assets/70551445/5e292990-62a6-488c-aee1-5884a3b855ba

### The demonstration of the interactive 3D visual model (ankle)
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
