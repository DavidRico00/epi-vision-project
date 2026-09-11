import cv2

def redimensionar_proporcional(imagen, ancho_deseado):
    alto_ori, ancho_ori = imagen.shape[:2]
    factor = ancho_deseado / ancho_ori
    alto_nuevo = int(alto_ori * factor)

    redimensinada = cv2.resize(imagen, (ancho_deseado, alto_nuevo))
    cv2.imwrite(f"ejercicios/modulo_01/output/resize_{ancho_deseado}.png", redimensinada)

    print(f"Nuevo shape: {redimensinada.shape}")

if __name__ == "__main__":
    ruta = "data/images/hard_hat_workers0.png"
    imagen = cv2.imread(ruta)

    redimensionar_proporcional(imagen, 200)

