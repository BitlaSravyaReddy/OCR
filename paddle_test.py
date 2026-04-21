import os
os.environ["FLAGS_use_mkldnn"] = "0"
from paddleocr import PaddleOCR
import cv2
import numpy as np

from PIL import Image
img = cv2.imread("images/sample6.jpg")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, 2
)
# Save preprocessed image
cv2.imwrite("processed6.jpg", thresh)
ocr = PaddleOCR(
    lang='hi',  # IMPORTANT for Hindi
    use_textline_orientation=True  # new replacement
    )

# Run OCR inference on a sample image 
result = ocr.predict(
    input="processed6.jpg")

# Visualize the results and save the JSON results
texts = result[0]['rec_texts']
boxes = result[0]['rec_polys']

for text, box in zip(texts, boxes):
    print(text)

    pts = [(int(x), int(y)) for x, y in box]
    cv2.polylines(img, [np.array(pts)], True, (0, 255, 0), 2)

    cv2.putText(img, text, pts[0],
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
cv2.imwrite("output6.jpg", img)

