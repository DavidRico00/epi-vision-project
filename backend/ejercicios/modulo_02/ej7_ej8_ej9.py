import cv2
import numpy as np

imagen = cv2.imread("data/images/hard_hat_workers0.png")
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
_, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
rango_bajo = np.array([15, 100, 100])
rango_alto = np.array([35, 255, 255])
mascara_casco = cv2.inRange(imagen_hsv, rango_bajo, rango_alto)


# --- Ejercicio 1: erosión vs dilatación sobre la binaria (Otsu) ---

kernel_5 = np.ones((5, 5), np.uint8)

erosionada = cv2.erode(binaria, kernel_5, iterations=1)
dilatada = cv2.dilate(binaria, kernel_5, iterations=1)

cv2.imwrite("ejercicios/modulo_02/output/ej1_erosion.png", erosionada)
cv2.imwrite("ejercicios/modulo_02/output/ej1_dilatacion.png", dilatada)

# Comparación numérica del tamaño de las regiones blancas (nº de píxeles blancos)
print(f"Píxeles blancos original:   {cv2.countNonZero(binaria)}")
print(f"Píxeles blancos erosión:    {cv2.countNonZero(erosionada)}")
print(f"Píxeles blancos dilatación: {cv2.countNonZero(dilatada)}")


# --- Ejercicio 2: cierre + apertura sobre la máscara de color del casco ---

mascara_cierre = cv2.morphologyEx(mascara_casco, cv2.MORPH_CLOSE, kernel_5)
mascara_cierre_apertura = cv2.morphologyEx(mascara_cierre, cv2.MORPH_OPEN, kernel_5)

cv2.imwrite("ejercicios/modulo_02/output/ej2_mascara_original.png", mascara_casco)
cv2.imwrite("ejercicios/modulo_02/output/ej2_mascara_cierre.png", mascara_cierre)
cv2.imwrite("ejercicios/modulo_02/output/ej2_mascara_cierre_apertura.png", mascara_cierre_apertura)


# --- Ejercicio 3: mismo proceso variando tamaño de kernel ---

tamanos_kernel = [3, 5, 7, 11]

for tam in tamanos_kernel:
    kernel_variable = np.ones((tam, tam), np.uint8)

    resultado = cv2.morphologyEx(mascara_casco, cv2.MORPH_CLOSE, kernel_variable)
    resultado = cv2.morphologyEx(resultado, cv2.MORPH_OPEN, kernel_variable)

    nombre_archivo = f"ejercicios/modulo_02/output/ej3_kernel_{tam}x{tam}.png"
    cv2.imwrite(nombre_archivo, resultado)
    print(f"Kernel {tam}x{tam} -> píxeles blancos: {cv2.countNonZero(resultado)}")