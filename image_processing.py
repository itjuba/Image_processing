import cv2 as cv
import sys
import numpy as np
from matplotlib import pyplot as plt



   
   
def image_processing (kernel_value,url="test1.jpg"):#image1 and image3 gave the best results.
		
	image = cv.imread(url)	
	
	sharpen_kernel =  np.array([[-1,-1,-1], [-1,kernel_value,-1], [-1,-1,-1]])#  creating 3*3 sharpening kernel
	sharpen = cv.filter2D(image, -1, sharpen_kernel) # applying the kernel 
	dst = cv.fastNlMeansDenoising(sharpen,None,12,7,21)#applying non local denoising
	
	 #image resizing for displaying purpose only 
	resized_sharp =  cv.resize(sharpen, (600, 400))  
	resized_orginal =  cv.resize(image, (600, 400))   
	resized_noice_removed =  cv.resize(dst, (600, 400))   


	# show time ....
	cv.imshow("original image", resized_orginal)
	cv.imshow("Sharpened image", resized_sharp)
	cv.imshow("noise remove image",resized_noice_removed)


	

	k = cv.waitKey(0)
	if k == ord("s"):
	    cv2.destroyAllWindows()
		
		
image_processing(9)
				
	
	
