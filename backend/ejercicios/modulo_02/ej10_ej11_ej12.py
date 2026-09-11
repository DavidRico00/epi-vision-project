import cv2
import numpy as np

imagen = cv2.imread("data/images/hard_hat_workers0.png")
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
_, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel_5 = np.ones((5, 5), np.uint8)


# --- Ejercicio 1: contornos sobre la binaria (Otsu) ---

contornos_otsu, _ = cv2.findContours(
    binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

imagen_contornos = imagen.copy()
cv2.drawContours(imagen_contornos, contornos_otsu, -1, (0, 255, 0), 2)

print(f"Ej1 - Contornos encontrados sobre Otsu: {len(contornos_otsu)}")
cv2.imwrite("ejercicios/modulo_02/output/ej1_contornos_otsu.png", imagen_contornos)


# --- Ejercicio 2: contornos sobre la máscara de color, filtrados por área ---

imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

# Ajusta este rango con el HSV real leído en el ejercicio anterior
rango_bajo = np.array([15, 80, 80])
rango_alto = np.array([35, 255, 255])
mascara_casco = cv2.inRange(imagen_hsv, rango_bajo, rango_alto)

mascara_limpia = cv2.morphologyEx(mascara_casco, cv2.MORPH_CLOSE, kernel_5)
mascara_limpia = cv2.morphologyEx(mascara_limpia, cv2.MORPH_OPEN, kernel_5)

contornos_color, _ = cv2.findContours(
    mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

imagen_boxes = imagen.copy()
contornos_validos = 0

for contorno in contornos_color:
    area = cv2.contourArea(contorno)
    if area < 200:
        continue
    contornos_validos += 1
    x, y, ancho, alto = cv2.boundingRect(contorno)
    cv2.rectangle(imagen_boxes, (x, y), (x + ancho, y + alto), (0, 255, 0), 2)

print(f"Ej2 - Contornos válidos (área > 200) en máscara de color: {contornos_validos}")
cv2.imwrite("ejercicios/modulo_02/output/ej2_boxes_color.png", imagen_boxes)


# --- Ejercicio 3: pipeline completo sobre varias imágenes ---

def pipeline_clasico(ruta_imagen, rango_bajo, rango_alto):
    img = cv2.imread(ruta_imagen)
    if img is None:
        print(f"No se pudo leer: {ruta_imagen}")
        return None, []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(hsv, rango_bajo, rango_alto)

    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel_5)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel_5)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    resultado_img = img.copy()
    detecciones = []

    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area < 200:
            continue

        perimetro = cv2.arcLength(contorno, True)
        if perimetro == 0:
            continue
        circularidad = (4 * np.pi * area) / (perimetro ** 2)

        x, y, ancho, alto = cv2.boundingRect(contorno)
        relacion_aspecto = ancho / float(alto)

        detecciones.append({
            "area": area,
            "circularidad": circularidad,
            "aspecto": relacion_aspecto,
            "bbox": (x, y, ancho, alto)
        })
        cv2.rectangle(resultado_img, (x, y), (x + ancho, y + alto), (0, 255, 0), 2)

    return resultado_img, detecciones


rutas_prueba = [
    "data/images/hard_hat_workers0.png",
    "data/images/hard_hat_workers1.png",
    "data/images/hard_hat_workers2.png",
    "data/images/hard_hat_workers3.png",
    # añade aquí rutas reales de tu dataset, idealmente con cascos/luz distintos
]

for ruta in rutas_prueba:
    resultado, detecciones = pipeline_clasico(ruta, rango_bajo, rango_alto)
    if resultado is None:
        continue

    nombre_base = ruta.split("/")[-1].replace(".png", "")
    cv2.imwrite(f"ejercicios/modulo_02/output/ej3_{nombre_base}_pipeline.png", resultado)

    print(f"\n{nombre_base} -> {len(detecciones)} detección(es) tras filtro de área:")
    for d in detecciones:
        print(f"  area={d['area']:.0f}, circularidad={d['circularidad']:.2f}, "
              f"aspecto={d['aspecto']:.2f}, bbox={d['bbox']}")