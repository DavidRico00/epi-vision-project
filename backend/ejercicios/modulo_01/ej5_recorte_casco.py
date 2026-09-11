import ej4_leer_xml as EJ4
import cv2
import os

ruta_imagen = "data/images/hard_hat_workers0.png"
imagen = cv2.imread(ruta_imagen)

detecciones = EJ4.leer_anotaciones("data/annotations/hard_hat_workers0.xml")
cascos = [d for d in detecciones  if(d["clase"] == "helmet")]

primer_casco = cascos[0]

recorte = imagen[primer_casco["ymin"]:primer_casco["ymax"], primer_casco["xmin"]:primer_casco["xmax"]]

os.makedirs("ejercicios/modulo_01/output", exist_ok=True)

cv2.imwrite("ejercicios/modulo_01/output/mi_imagen.png", recorte)