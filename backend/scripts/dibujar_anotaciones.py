import cv2
import xml.etree.ElementTree as ET

ruta_imagen = "data/images/hard_hat_workers0.png"
ruta_anotacion = "data/annotations/hard_hat_workers0.xml"

imagen = cv2.imread(ruta_imagen)

# Parseamos el XML (formato Pascal VOC)
tree = ET.parse(ruta_anotacion)
root = tree.getroot()

# Colores distintos por clase (BGR, recuerda: OpenCV usa BGR no RGB)
colores = {
    "helmet": (0, 255, 0),   # verde
    "head": (0, 0, 255),     # rojo
    "person": (255, 0, 0),   # azul
}

for objeto in root.findall("object"):
    clase = objeto.find("name").text
    bbox = objeto.find("bndbox")
    xmin = int(bbox.find("xmin").text)
    ymin = int(bbox.find("ymin").text)
    xmax = int(bbox.find("xmax").text)
    ymax = int(bbox.find("ymax").text)

    color = colores.get(clase, (255, 255, 255))
    cv2.rectangle(imagen, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.putText(imagen, clase, (xmin, ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    print(f"Clase: {clase} | Caja: ({xmin},{ymin}) - ({xmax},{ymax})")

cv2.imshow("Anotaciones - hard_hat_workers0", imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()