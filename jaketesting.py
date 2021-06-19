import numpy as np
import cv2

s1 = cv2.imread("Images/1_colour.jpeg", 1) #Scene with 1 object
s2 = cv2.imread("Images/2_colour.jpeg") #Scene with 2 objects
mask = cv2.imread("Images/masks/souvenirs_no_3_colour_mask_2_mask.png") #Target object mask

def ExtractObject(S2, ObjectMask):
    """Extracts an object from an image scene based on the mask used"""

    final_im = ObjectMask*S2
    final_im = cv2.bitwise_not(final_im)
    return final_im

def ApplyFilter(ExtractedObject, FilterIndex):
    """Applies convolution on the object using pre-defined image kernels"""

    if FilterIndex==0: #Apply no filter
        FilteredExObject = ExtractedObject
    elif FilterIndex==1:
        kernel = np.ones((5, 5), np.float32)/25
        FilteredExObject = cv2.filter2D(ExtractedObject, -1, kernel)
        #Will need to define some kernels and use them here
    elif FilterIndex==2:
        img_yuv = cv2.cvtColor(ExtractedObject, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        FilteredExObject = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    elif FilterIndex==3:
        FilteredExObject = cv2.medianBlur(ExtractedObject, 5)
        #Will need to define some kernels and use them here
    return FilteredExObject

def ObjectBlender(S1, FilteredExObject):
    """Adds the filtered extracted object to the image in scene S1"""

    alpha = 0.6
    beta = 1.0 - alpha
    BlendingResult = cv2.addWeighted(S1, alpha, FilteredExObject, beta, -60)
    cv2.imshow('Blended', BlendingResult)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return BlendingResult

def CompareResult(BlendingResult, S2, metric):
    """Compares the blended image with the scene with 2 objects"""

    if metric==1: #Sum of Squared Distance Error (SSD)
        error = np.sum((BlendingResult-S2)**2)
    elif metric==2: #Mean Squared Error (MSE)
        error = np.sum((BlendingResult.astype("float") - S2.astype("float")) ** 2)
        error /= float(BlendingResult.shape[0] * BlendingResult.shape[1])
    return error

extracted = ExtractObject(s2, mask)
extracted = ApplyFilter(extracted, 3)

blended = ObjectBlender(s1, extracted)
print(CompareResult(blended, s2, 2))

def NewBackground(imgNoBg, NewBackground):
    """Replaces the background of an image with NewBackground"""
    ObjectBlender(imgNoBg, NewBackground)

newb = cv2.imread("gorg.jpg")
newb = cv2.resize(newb, (extracted.shape[1], extracted.shape[0]))
NewBackground(extracted, newb)
    