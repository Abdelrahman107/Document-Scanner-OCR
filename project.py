#Importing necessary libraries
from  cv2 import cv2
import re
import pytesseract
import numpy as np
from imutils.perspective import four_point_transform
width = 800
height = 800

def convert_to_grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def apply_canny(img,low_thres,high_thres):
    """
    Canny steps:
    1.Apply gaussian blur
    2.image gradient
    3.non-maximum supression
    4.Hysteresis Thresholding
    """
    blur = cv2.GaussianBlur(img,(5,5),0)
    egdes = cv2.Canny(blur,low_thres,high_thres)
    return egdes
    # to do make canny more adaptive.


#Due to thining-- closing is used for cutout pixels
def apply_closingMorphlogy(img,SE):
    kernel = np.ones((SE,SE),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel,iterations=1)


def find_largest_contours(img,imagecopy):
    """
    Steps of this function:
    1. find all contours from the edged image
    2. sort contours on area descendingly
    3. get the first 5 maximum contours
    4. for each contour, approxiamate egde points to only four points to targer square contour
    """
    cimg = img.copy()
    contours,hierarchy = cv2.findContours(cimg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    #draw_contours(imagecopy,cnts)
    #showImage(imagecopy)
    

    for c in cnts:
        perimeter = cv2.arcLength(c,True)
        # only targeting squares or rectanges
        approximation = cv2.approxPolyDP(c,0.02*perimeter,True)
        if  len(approximation)==4: 
            target = approximation
            break
        else:
            return cimg 
    return target
    
    
               
 

def draw_contours(img,contours):
    cv2.drawContours(img, contours, -1, (0,255,0), 50)


def showImage(img):
    cv2.imshow('Output Image',img)
    cv2.waitKey(0)



def resize_image(img):
    resize = cv2.resize(img, (width, height))
    return resize

def transform(points_contour,img):
     
    if points_contour.size != 0  and points_contour.size ==8:
        ordered_points = order(points_contour)
        list_pts1 = np.float32(ordered_points)
 
        """
        # an attempt to improve the algorithm by computing euclidean distance for generic code
        (tl, tr, br, bl) = list_pts1
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

       
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))

        """

        list_pts2 = np.float32([[0, 0],[width, 0], [0, height],[width, height]]) 
        M = cv2.getPerspectiveTransform(list_pts1, list_pts2)
        warp = cv2.warpPerspective(img, M, (width,height))
        return warp
    else:
        return img
        




    
def order(contour_points):
    """
    Steps for this fucntion: 
    1. parameter given is four points of the largest  contour
    2.  inialize a list of four points --> ordered_contour_points
    3. this list will have top left, top right, bottom right, bottom left 
    4. list[0] will have the point of minmum sum, list list[3] maximum sum.. etc
    """
    
    contour_points = contour_points.reshape((4, 2))
    ordered_contour_points = np.zeros((4, 1, 2), dtype=np.int32)
    add = contour_points.sum(1)

    ordered_contour_points[0] = contour_points[np.argmin(add)]
    ordered_contour_points[3] =contour_points[np.argmax(add)]
    diff = np.diff(contour_points, axis=1)
    ordered_contour_points[1] =contour_points[np.argmin(diff)]
    ordered_contour_points[2] = contour_points[np.argmax(diff)]
    return ordered_contour_points



def post_process(transformed_image):
    """
    Post processing if required. in most casses, it's not required.
    gray = cv2.cvtColor(transformed_image,cv2.COLOR_BGR2GRAY)
    
    
    #blur=cv2.medianBlur(gray,5)
    #thres0= cv2.adaptiveThreshold(blur, 255, 1, 1, 7, 1)
    #rat,thres=cv2.threshold(blur, 0, 100, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #showImage(thres)
    #thres=cv2.erode(thres,np.ones((3, 3), np.uint8))
    #howImage(thres)
    #thres=cv2.dilate(thres,np.ones((1, 1), np.uint8))
    #showImage(thres)

    
    imgAdaptiveThre = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,51,2)

    imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
    imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,5)
    #showImage(imgAdaptiveThre)

    """
    return transformed_image


def count_characters(text):
    return (len(re.sub(r"\W", "", text)))



def Preprocessing_hard_casses(img):
    gray = convert_to_grayscale(img)
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
    thresh = cv2.Canny(sharpen,160,200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    cv2.imwrite('D:\Courses\Computer Vision\Project\TempFolder\gray.jpg',gray)
    cv2.imwrite('D:\Courses\Computer Vision\Project\TempFolder\ blur.jpg',blur)
    cv2.imwrite('D:\Courses\Computer Vision\Project\TempFolder\sharpen.jpg',sharpen)
    cv2.imwrite('D:\Courses\Computer Vision\Project\TempFolder\canny.jpg',thresh)
    cv2.imwrite('D:\Courses\Computer Vision\Project\TempFolder\close.jpg',close)
 
    
    
    return close

#---------------------------------
# Steps
#--------------------------------

#-------------------------------
# Main program
#-------------------------------


#reading image
image = cv2.imread("test10.jpg")
imagecopy=image.copy()
"""
#Old Preprocessing.
#converting to grayscale
#gray = convert_to_grayscale(image) 
#apply canny to detect edges
#canny = apply_canny(gray,75,200)
#apply dilating then eroson to connect cutoff pixels
#canny_Morph=apply_closingMorphlogy(canny,1)

"""

# new Preprocessing for hard casses images
canny_Morph= Preprocessing_hard_casses(image)


#detect contours in the image resulted from canny
contour_points = find_largest_contours(canny_Morph,imagecopy)
#draw_contours(imagecopy,contour_points)
#cv2.imwrite('D:\Courses\Computer Vision\Project\TempFolder\contour_detected.jpg',imagecopy)



#Prespective allignment and transformation
transformed_image = transform(contour_points,image)
#showImage(transformed_image)
cv2.imwrite('D:\Courses\Computer Vision\Project\TempFolder\ transformed.jpg',transformed_image)
 
#post-prcessing
final_image = post_process(transformed_image)


#OCR
text = pytesseract.image_to_string(final_image) 
print(text)
file = open("text.txt", "a") 
file.write(text) 
file.write("\n")
file.write("The number of characters detected without spaces are = ")
file.write(str(count_characters(text)))
file.close 


 



 