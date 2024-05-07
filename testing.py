import cv2
import pytesseract
import re
import numpy as np
import imutils
from skimage import exposure

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

harcascade = "model/numberplate_haarcade.xml"

cap = cv2.VideoCapture(0) # set an id if multiple cameras id=0 for default camera
cap.set(3, 640) #width
cap.set(4, 480) #height

min_area = 500
count = 0

# Create a set of authorized license plates
authorized_plates = {"MH20EE7602", "KA01AB1234", "DL3CAX1234"} # Add more plates as needed

access_granted = False

while True:
    success, img = cap.read() # returns sucess statement and reading image
    plate_cascade = cv2.CascadeClassifier(harcascade) # using harcascade xml file to detect object from image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Harcascade need grayscale image to deetct

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)
    for (x,y,w,h) in plates:
        area = w * h

        if area > min_area:                      #R,G,B
            cv2.rectangle(img, (x,y), (x+w, y+h),(0,255,0),2) # img, min_coordinate, max_coordinate, green color, thickness
            cv2.putText(img, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255), 2)
            # image cropping
            img_roi = img[y: y+h, x: x+w]
            cv2.imshow("ROI", img_roi)

            # Enhance the license plate image
            enhanced_img = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
            
            # Contrast stretching
            p2, p98 = np.percentile(enhanced_img, (2, 98))
            enhanced_img = exposure.rescale_intensity(enhanced_img, in_range=(p2, p98))
            
            # Histogram equalization
            enhanced_img = cv2.equalizeHist(enhanced_img)

            # Use pytesseract to extract text from the license plate
            text = pytesseract.image_to_string(enhanced_img)

            # Use regex to filter out irrelevant characters
            license_plate = re.sub('[^A-Z0-9]', '', text)
            print(license_plate)

            # Check if the recognized license plate is authorized
            if license_plate in authorized_plates:
                print("Access granted")
                access_granted = True
                break
            else:
                print("Access denied")

    if access_granted:
        break

    cv2.imshow("Result", img)
 
    if cv2.waitKey(1) & 0xFF == ord('q'): # QUIT THE CAMERA WINDOW BY PRESSING Q
        break
