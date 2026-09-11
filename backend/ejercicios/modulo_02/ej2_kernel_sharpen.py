import cv2
import numpy as np

kernel_sharpen_5 = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

kernel_sharpen_9 = np.array([
    [0, -2, 0],
    [-2, 9, -2],
    [0, -2, 0]
])

imagen = cv2.imread("data/images/hard_hat_workers0.png")

imagen_sharpen_5 = cv2.filter2D(imagen, -1, kernel_sharpen_5)
imagen_sharpen_9 = cv2.filter2D(imagen, -1, kernel_sharpen_9)

cv2.imwrite("ejercicios/modulo_02/output/imagen_sharpen_5.png", imagen_sharpen_5)
cv2.imwrite("ejercicios/modulo_02/output/imagen_sharpen_9.png", imagen_sharpen_9)