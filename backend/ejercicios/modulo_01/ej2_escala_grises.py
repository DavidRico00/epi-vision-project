import cv2

ruta_img = "data/images/hard_hat_workers0.png"
imagen = cv2.imread(ruta_img)

alto, ancho, canales = imagen.shape
print(f"Antiguo shape: {imagen.shape}")
print(f"Pixel central antiguo: {imagen[alto//2 , ancho//2]}")

imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

alto_gris, ancho_gris = imagen_gris.shape

print(f"Nuevo shape: {imagen_gris.shape}")
print(f"Pixel central nuevo: {imagen_gris[alto//2 , ancho//2]}")