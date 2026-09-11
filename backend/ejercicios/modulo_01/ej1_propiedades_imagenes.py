import cv2

ruta_imagen = "data/images/hard_hat_workers0.png"
imagen = cv2.imread(ruta_imagen)

if imagen is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen en: {ruta_imagen}")

alto, ancho, canales = imagen.shape
print(f"Dimension (ancho x alto): {ancho} x {alto}")
print(f"Canales de color: {canales}")
print(f"Tipo de dato: {imagen.dtype}")
print(f"Shape completa: {imagen.shape}")

print(f"Valor pixel esquina superior izquierda: {imagen[0,0]}")
print(f"Valor pixel central: {imagen[alto//2, ancho//2]}")