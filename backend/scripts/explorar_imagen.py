import cv2

# Cargamos una imagen del dataset
ruta_imagen = "data/images/hard_hat_workers0.png"
imagen = cv2.imread(ruta_imagen)

if imagen is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen en: {ruta_imagen}")

# Propiedades básicas del array de NumPy que representa la imagen
alto, ancho, canales = imagen.shape
print(f"Dimensiones (alto x ancho): {alto} x {ancho}")
print(f"Canales de color: {canales}")
print(f"Tipo de dato: {imagen.dtype}")
print(f"Shape completo: {imagen.shape}")

# Mostramos la imagen en una ventana
cv2.imshow("Hard Hat Workers - imagen 0", imagen)
print("Pulsa cualquier tecla sobre la ventana de la imagen para cerrarla...")
cv2.waitKey(0)
cv2.destroyAllWindows()