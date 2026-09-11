import cv2

def info_imagen(ruta):
    imagen = cv2.imread(ruta)

    if imagen is None:
        raise FileNotFoundError(f"No se encuentra el archivo en: {ruta}")

    es_color = len(imagen.shape) == 3

    if es_color:
        alto, ancho, canales = imagen.shape
    else:
        alto, ancho = imagen.shape
        canales = 1

    return{
        "alto": alto,
        "ancho": ancho,
        "canales": canales,
        "tipo_dato": str(imagen.dtype),
        "es_color": es_color,
    }

rutas = [
    "data/images/hard_hat_workers1.png",
    "data/images/hard_hat_workers2.png",
    "data/images/hard_hat_workers3.png",
]

for ruta in rutas:
    print(info_imagen(ruta))