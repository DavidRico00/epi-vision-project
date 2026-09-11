import cv2

imagen = cv2.imread("data/images/hard_hat_workers0.png")
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# --- Ejercicio 1: umbralización fija vs Otsu vs adaptativa ---

_, umbral_fijo = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)

umbral_otsu_valor, umbral_otsu = cv2.threshold(
    gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
print(f"Umbral calculado por Otsu: {umbral_otsu_valor}")

umbral_adaptativo = cv2.adaptiveThreshold(
    gris, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, 2
)

cv2.imwrite("ejercicios/modulo_02/output/ej1_umbral_fijo.png", umbral_fijo)
cv2.imwrite("ejercicios/modulo_02/output/ej1_umbral_otsu.png", umbral_otsu)
cv2.imwrite("ejercicios/modulo_02/output/ej1_umbral_adaptativo.png", umbral_adaptativo)


# --- Ejercicio 2: Canny con distintos pares de umbrales ---

gris_suavizado = cv2.GaussianBlur(gris, (5, 5), 0)

pares_umbrales = [(30, 90), (50, 150), (100, 200)]

for umbral_inf, umbral_sup in pares_umbrales:
    bordes = cv2.Canny(gris_suavizado, umbral_inf, umbral_sup)
    nombre_archivo = f"ejercicios/modulo_02/output/ej2_canny_{umbral_inf}_{umbral_sup}.png"
    cv2.imwrite(nombre_archivo, bordes)


# --- Ejercicio 3: comparar Otsu vs adaptativa en varias imágenes ---

rutas_imagenes = [
    "data/images/hard_hat_workers0.png",
    "data/images/hard_hat_workers1.png",
    "data/images/hard_hat_workers2.png",
]

for ruta in rutas_imagenes:
    img = cv2.imread(ruta)
    if img is None:
        print(f"No se pudo leer: {ruta}")
        continue

    gris_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nombre_base = ruta.split("/")[-1].replace(".png", "")

    _, otsu_img = cv2.threshold(
        gris_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    adaptativa_img = cv2.adaptiveThreshold(
        gris_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    cv2.imwrite(f"ejercicios/modulo_02/output/ej3_{nombre_base}_otsu.png", otsu_img)
    cv2.imwrite(f"ejercicios/modulo_02/output/ej3_{nombre_base}_adaptativa.png", adaptativa_img)