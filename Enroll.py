import cv2
import numpy as np

def iriscode(img):
    # Conversione in scala di grigi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Filtraggio dell'immagine con un filtro di Gabor
    kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
    # Segmentazione dell'iride
    _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    iris_contour = max(contours, key=cv2.contourArea)
    # Normalizzazione dell'iride
    (x, y), r = cv2.minEnclosingCircle(iris_contour)
    r = int(r)
    M = cv2.moments(iris_contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    normalized = cv2.normalize(filtered[cY-r:cY+r, cX-r:cX+r], None, alpha=0, beta=255,
    norm_type=cv2.NORM_MINMAX)
    # Codifica dell'iride con IrisCode
    iriscode = ""
    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            if normalized[i][j] > normalized.mean():
                iriscode += "1"
            else:
                iriscode += "0"
    return iriscode
