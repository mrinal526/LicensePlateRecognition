import cv2
import easyocr
import re

# Initialize cascade classifier
numberPlate_cascade = "model/numberplate_haarcade.xml"  # Assuming "model" folder is in the same directory
detector = cv2.CascadeClassifier(numberPlate_cascade)

# Initialize the easyocr Reader object
reader = easyocr.Reader(['en'])

# Target aspect ratio for resizing ROI (adjust as needed)
target_aspect_ratio = 3/4

def preprocess_roi(roi):
    # Apply bilateral filter for noise reduction
    roi = cv2.bilateralFilter(roi, 9, 75, 75)
    return roi

def binarize_roi(roi):
    thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

def apply_morphology(thresh):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return opening

def resize_roi(roi, target_aspect_ratio):
    height, width = roi.shape[:2]
    new_width = int(height * target_aspect_ratio)
    resized_roi = cv2.resize(roi, (new_width, height), interpolation=cv2.INTER_AREA)
    return resized_roi

def preprocess_and_recognize_plate(plate_roi, reader):
    preprocessed_roi = preprocess_roi(plate_roi)
    binarized_roi = binarize_roi(preprocessed_roi)
    morphologically_refined_roi = apply_morphology(binarized_roi)
    resized_roi = resize_roi(morphologically_refined_roi, target_aspect_ratio)
    text = reader.readtext(resized_roi)
    return text

def is_valid_indian_plate(text):
    pattern = r"^[A-Z]{2}\d{2}[A-Z]{1}\d{4}$"
    return bool(re.match(pattern, text))

# Create a VideoCapture object to access the webcam
cap = cv2.VideoCapture(0)  # 0 for the default webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is captured successfully
    if not ret:
        print("Error: Unable to capture frame from webcam")
        break

    # Process the frame (convert to grayscale, detect plates, etc.)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # *Adjust parameters here (if needed):*
    plates = detector.detectMultiScale(gray, 1.1, 5)  # Example: increased scaleFactor, decreased minNeighbors

    for (x, y, w, h) in plates:
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the number plate
        plate_roi = gray[y:y + h, x:x + w]

        # Preprocess and recognize the plate
        text = preprocess_and_recognize_plate(plate_roi, reader)

        if len(text) > 0:
            print(text)

            # # Optional: Check for valid Indian format
            # if is_valid_indian_plate(text[0][1]):
            #     print("Valid Indian number plate format")
            # else:
            #     print("Warning: Plate format might not be valid")

            # Draw text in the frame
            cv2.putText(frame, text[0][1], (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Number Plate Recognition', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
# cap.release()
# cv2.destroyAllWindows()