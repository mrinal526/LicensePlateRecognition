import cv2
import pytesseract
import re
import numpy as np
import imutils
from skimage import exposure

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# harcascade = "model/numberplate_haarcade.xml"

cap = cv2.VideoCapture(0) # set an id if multiple cameras id=0 for default camera
cap.set(3, 640) #width
cap.set(4, 480) #height

min_area = 500
# count = 0

# Create a set of authorized license plates
authorized_plates = {"MH20EE7602", "KA01AB1234", "DL3CAX1234"} # Add more plates as needed

access_granted = False

while True:
    success, frame = cap.read() # returns sucess statement and reading image
    if not success:
        break

    # cv2.imshow("Frame", frame)
    # key = cv2.waitKey(1) & 0xFF
    
    # plate_cascade = cv2.CascadeClassifier(harcascade) # using harcascade xml file to detect object from image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None

    # plates = plate_cascade.detectMultiScale(gray, 1.1, 4)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        print("")
    else:
        cv2.drawContours(frame, [screenCnt], -1, (0, 255, 0), 3)
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
        new_image = cv2.bitwise_and(frame, frame, mask=mask)
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]
        text = pytesseract.image_to_string(Cropped, config='--psm 11')
        # print("Detected Number is:", text)
        cv2.imshow('Cropped', Cropped)
        # Use regex to filter out irrelevant characters
        license_plate = re.sub('[^A-Z0-9]', '', text)
        print(license_plate)

        # Check if the recognized license plate is authorized
        if license_plate in authorized_plates:
            print("Authorized")
            access_granted = True
            break
        else:
            print("Not Authorized")
        

    if access_granted:
        break

    cv2.imshow("Result", frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'): # QUIT THE CAMERA WINDOW BY PRESSING Q
        break
