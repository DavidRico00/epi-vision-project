import xml.etree.ElementTree as ET

def leer_anotaciones(ruta_xml):
    tree = ET.parse(ruta_xml)
    root = tree.getroot()

    detecciones = []

    for objeto in root.findall("object"):
        bndbox = objeto.find("bndbox")

        detecciones.append({
            "clase": objeto.find("name").text,
            "xmin": int(bndbox.find("xmin").text),
            "ymin": int(bndbox.find("ymin").text),
            "xmax": int(bndbox.find("xmax").text),
            "ymax": int(bndbox.find("ymax").text),
        })

    return detecciones


if __name__ == "__main__":
    print(leer_anotaciones("data/annotations/hard_hat_workers0.xml"))