import cv2
import numpy as np
import os

# Ensure output directory exists
output_dir = "content/extract"
os.makedirs(output_dir, exist_ok=True)

# Read the image
img = cv2.imread('content/seg/img.png')
rsz_img = cv2.resize(img, (1024, 1024))  # Resize image to 1024x1024
gray = cv2.cvtColor(rsz_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

img_2 = cv2.imread('content/seg/24RR000004-G-1-1_11520_28416.jpeg')
rsz_img_2 = cv2.resize(img_2, None, fx=1, fy=1)

# Threshold the image
retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Process each contour
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)  # Get bounding rectangle
    x, y, w, h = x - 10, y - 10, w + 20, h + 20  # Expand the rectangle

    # Crop the region from the original RGB image (img_2)
    cropped_region = rsz_img_2[y:y + h, x:x + w]

    # Save the cropped region to the output directory
    output_path = os.path.join(output_dir, f"cropped_{i + 1}.png")
    cv2.imwrite(output_path, cropped_region)

    print(f"Saved cropped image {i + 1} to {output_path}")

cv2.destroyAllWindows()