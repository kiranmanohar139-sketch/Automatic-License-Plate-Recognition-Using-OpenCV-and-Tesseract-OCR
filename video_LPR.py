import cv2
import pytesseract
import imutils

# Path to Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load video instead of webcam
cap = cv2.VideoCapture("car_video.mp4")   # <-- put your video filename here

if not cap.isOpened():
    print("❌ Cannot open video file")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break  # end of video

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(c)
            plate = gray[y:y+h, x:x+w]
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
            text = pytesseract.image_to_string(plate, config='--psm 8')
            cv2.putText(frame, text.strip(), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            break

    cv2.imshow("License Plate Recognition (Video)", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()