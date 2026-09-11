import cv2
import os

imagen = cv2.imread("data/images/hard_hat_workers0.png")

imagen_blur = cv2.blur(imagen, (5, 5))
imagen_gaus = cv2.GaussianBlur(imagen, (5, 5), 0)
imagen_median = cv2.medianBlur(imagen, 5)

os.makedirs("ejercicios/modulo_02/output", exist_ok=True)

cv2.imwrite("ejercicios/modulo_02/output/imagen_blur.png", imagen_blur)
cv2.imwrite("ejercicios/modulo_02/output/imagen_gaus.png", imagen_gaus)
cv2.imwrite("ejercicios/modulo_02/output/imagen_median.png", imagen_median)
