import cv2
import imutils
import numpy as np
import pytesseract
import re
# import mysql.connector

# Connect to MySQL database
# mydb = mysql.connector.connect(
#     host="localhost",
#     user="yourusername",
#     password="yourpassword",
#     database="yourdatabase"
# )

# mycursor = mydb.cursor()
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
harcascade = "model/numberplate_haarcade.xml"

# Initialize webcam
camera = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    if not ret:
        break


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 30, 200)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        screenCnt = None

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break

        if screenCnt is None:
            print("No contour detected")
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
            license_plate = re.sub('[^A-Z0-9]', '', text)
            print("Detected Number is:", license_plate)

            # Check if the detected number exists in the database
            # sql = "SELECT * FROM your_table WHERE number = %s"
            # val = (text,)
            # mycursor.execute(sql, val)
            # result = mycursor.fetchone()

            # Print authorized or not authorized based on database result
            # if result:
            #     print("Authorized")
            # else:
            #     print("Not Authorized")

            cv2.imshow('Cropped', Cropped)
            cv2.waitKey(0)
            break

# Release the camera and close OpenCV windows
camera.release()
cv2.destroyAllWindows()