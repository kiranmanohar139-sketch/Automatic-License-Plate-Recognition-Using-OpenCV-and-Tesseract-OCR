import cv2
import pytesseract
import numpy as np

# Set path for Tesseract (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load image
image_path = "car_image.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found.")
    exit()

print("Image loaded:", image.shape)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Noise removal
gray = cv2.bilateralFilter(gray, 11, 17, 17)

# Edge detection
edges = cv2.Canny(gray, 30, 200)

# Find contours
cnts, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Total contours found:", len(cnts))

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

plate_region = None

# Loop through contours
for i, c in enumerate(cnts):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)

    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)

        # Basic filtering for plate shape
        if w > 100 and h > 20:
            print("Plate candidate found at:", x, y, w, h)
            plate_region = gray[y:y + h, x:x + w]
            break

if plate_region is None:
    print("No plate region detected.")
    exit()

print("Plate region size:", plate_region.shape)

# OCR
text = pytesseract.image_to_string(
    plate_region,
    config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.'
)

# Clean OCR text
text = text.strip().replace(" ", "").replace("\n", "")

print("Detected Number Plate Text:", text)

# Show extracted plate
cv2.imshow("Plate Region", plate_region)
cv2.imshow("Original Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("Enhanced License Plate", thresh)

