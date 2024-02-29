import cv2
import numpy as np

# Re-define the function as the execution state has been reset
def segment_plate_chars(image_path):
    # Read image
    img = cv2.imread(image_path)

    # Pre-processing steps
    img1 = cv2.resize(img, (320, 100), interpolation=cv2.INTER_AREA)
    img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img3 = cv2.bilateralFilter(img2, 11, 17, 17)
    img4 = cv2.Canny(img3, 30, 200)  # Adjusted the thresholds for this image
    img5 = img4[10:90, 10:310]
    crop_img = img1[10:90, 10:310, :]

    # Find contours
    contours, _ = cv2.findContours(img5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 500 < w*h < 4000 and 20 < h < 80 and w < 80:
            candidate.append([x, (x + w)])

    # Based on contour positions, create a marker array
    loc = np.zeros(300)
    for j in range(len(candidate)):
        x1, x2 = candidate[j]
        loc[x1:x2] = 1

    # Find start and end positions of characters
    start, end = [], []
    if loc[0] == 1:
        start.append(0)
    for j in range(299):
        if loc[j] == 0 and loc[j+1] == 1:
            start.append(j+1)
        if loc[j] == 1 and loc[j+1] == 0:
            end.append(j)
    if loc[299] == 1:
        end.append(299)

    # Split characters
    char_images = []
    if len(start) == len(end) == 7:
        for j in range(7):
            x1, x2 = start[j], end[j]
            y1, y2 = 0, 80
            char_img = crop_img[y1:y2, x1:x2]
            char_images.append(char_img)
    return char_images

# Specify the image path and call the function
image_path = './carplate6.png'
char_images = segment_plate_chars(image_path)

# Save segmented character images and provide the paths list
output_paths = []
for i, char_img in enumerate(char_images):
    output_path = f'./datasets/char_{7+i}.png'
    cv2.imwrite(output_path, char_img)
    output_paths.append(output_path)