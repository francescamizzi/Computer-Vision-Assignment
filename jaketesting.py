import cv2, numpy as np

def ObjectBlender(s1, FilteredExObject):
    """Adds the filtered extracted object to the image in scene s1"""
    alpha = 0.5
    beta = 1 - alpha
    blendedObject = cv2.addWeighted(s1, alpha, FilteredExObject, beta, 0.0)
    cv2.imshow('Blended', blendedObject)
    cv2.waitKey(0)
    return blendedObject


img = cv2.imread("Images/1_colour.jpeg", 1)
object = cv2.imread("Images\souvenirs_no_3_colour_mask_2_mask.png", 1)
blended = ObjectBlender(img, object)