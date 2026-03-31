import cv2
import numpy as np

orgial_PCB = cv2.imread('template.jpg')
Tested_PCB = cv2.imread('defect.jpg')

gray_base = cv2.cvtColor(orgial_PCB, cv2.COLOR_BGR2GRAY)
gray_sample = cv2.cvtColor(Tested_PCB, cv2.COLOR_BGR2GRAY)
difference = cv2.absdiff(gray_base, gray_sample)

_, binary_result = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(binary_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    if cv2.contourArea(c) >5:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(Tested_PCB, (x, y), (x + w, y + h), (0 ,0 ,255), 2)

        cv2.imshow('Detection Result', Tested_PCB)
cv2.waitKey(0)
cv2.destroyAllWindows()
