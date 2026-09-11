import cv2
import numpy as np

imagen = cv2.imread("data/images/hard_hat_workers0.png")

kernel_casco = np.array([
    [-1, -1, -1, -1, -1],
    [-1,  2,  2,  2, -1],
    [-1,  2,  4,  2, -1],
    [-1,  2,  2,  2, -1],
    [-1, -1, -1, -1, -1]
])

resaltado = cv2.filter2D(imagen, -1, kernel_casco)
cv2.imwrite("ejercicios/modulo_02/output/imagen_resaltado_casco.png", resaltado)