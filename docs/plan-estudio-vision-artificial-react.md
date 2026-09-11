# 📘 Plan de Estudio: Visión Artificial (Computer Vision) y React

**Perfil del alumno:** Experiencia previa en programación general (no en CV ni React concretamente). No requiere explicación de conceptos básicos de programación (variables, bucles, funciones, POO), sino los conceptos, herramientas y particularidades específicas de cada materia.

**Objetivo:** Proyecto de trabajo que combina un backend de visión artificial (procesamiento/detección sobre imágenes) con una interfaz en React que consume esos resultados.

**Ritmo:** Intensivo — 8 semanas, compaginado con trabajo. Teoría concisa, ejemplos comentados, ejercicios progresivos (básico → intermedio → reto) por tema.

## 🎯 Proyecto único del curso

**Nombre:** Sistema de monitorización de EPI (cascos de seguridad) en entornos de trabajo.

**Qué hace:** a partir de una imagen o vídeo de una obra/planta, el sistema detecta a las personas presentes y determina si llevan puesto el casco de seguridad, mostrando el resultado en un dashboard web (React) con las detecciones marcadas sobre la imagen y un resumen de cumplimiento (nº de personas con/sin casco).

**Dataset público a usar:** [Hard Hat Detection (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection) — imágenes anotadas con las clases `helmet`, `head` (sin casco) y `person`, en formato compatible con frameworks de detección. Es un dataset consolidado y muy usado en proyectos similares de seguridad laboral, con volumen suficiente para practicar tanto inferencia con modelo preentrenado como fine-tuning ligero.

**Por qué este proyecto:** cubre de forma natural todo el temario — procesamiento de imagen clásico, detección de objetos con deep learning, backend que expone el modelo como API, y frontend en React que consume esa API — y tiene aplicación directa y explicable en un entorno de trabajo (compliance/seguridad), sin depender de datos privados de la empresa.

**Cómo evoluciona el mismo proyecto módulo a módulo:**
- Módulos 0-2: se trabaja directamente sobre imágenes del dataset (entorno, exploración, preprocesado).
- Módulo 3: base conceptual necesaria antes de aplicar el modelo de detección.
- Módulo 4: detección de cascos con YOLO sobre el dataset (preentrenado + ajuste/fine-tuning ligero a las clases del dataset).
- Módulo 5: el detector de cascos se expone como API propia con Django + Django REST Framework, con histórico de detecciones guardado en base de datos.
- Módulos 6-7: se construye el dashboard en React que consume esa API.
- Módulo 8: integración final del sistema completo y demo con casos variados del propio dataset.

**Metodología:**
- Un tema/sub-tema a la vez, nunca todo de golpe.
- 2-5 ejercicios por tema, con feedback antes de avanzar.
- Ejemplos contextualizados en el proyecto real del trabajo siempre que sea posible.
- Este archivo se actualizará marcando `[x]` según avancemos.
- **Al completar cada módulo**, además de marcar sus checkboxes, se debe añadir/actualizar una entrada correspondiente en la sección **"📚 Resumen de progreso por módulo"** al final de este archivo, con: qué se cubrió, qué código/proyectos se crearon (nombres y ubicación si aplica), y cualquier decisión o matiz relevante para retomar el hilo en un chat nuevo sin perder contexto.
- **Existe un archivo complementario `teoria-vision-react.md`** con toda la teoría y ejemplos ya explicados (sin ejercicios), organizado por módulo y sub-tema. Cada vez que se explique un tema o sub-tema nuevo, ese contenido debe incorporarse a ese archivo para que un chat nuevo sepa exactamente qué conceptos conoce ya el alumno.
  - **⚠️ Nota de flujo de trabajo (importante):** los archivos de este proyecto (`plan-estudio-vision-react.md` y `teoria-vision-react.md`) son copias de solo lectura dentro de la sesión de Claude — cualquier edición que Claude haga sobre ellos con sus herramientas **no se guarda de vuelta en el proyecto real** del alumno. Por eso, el flujo correcto es:
    1. Claude **sí** puede regenerar y entregar como archivo descargable la versión actualizada de `plan-estudio-vision-react.md` (checkboxes + resumen de progreso), ya que es un archivo relativamente corto y se sobreescribe entero sin problema.
    2. Para `teoria-vision-react.md` (archivo largo), al completar cada módulo Claude **entrega en el chat** (como texto, no como archivo aparte) únicamente la teoría nueva cubierta en ese módulo — el alumno la copia y la pega manualmente en su copia real del archivo, en la sección del módulo correspondiente.
    3. Si en algún momento se prefiere que Claude regenere el archivo `teoria-vision-react.md` completo como descarga (en vez de fragmentos en chat), basta con pedirlo explícitamente.

---

## Módulo 0 — Entorno y primer contacto
- [ ] Python: entorno virtual, gestión de dependencias (`pip`, `venv`/`conda`)
- [ ] Instalación de librerías clave: NumPy, OpenCV, Ultralytics (YOLO), Django, Django REST Framework
- [ ] Node.js/npm y creación de un proyecto React (Vite)
- [ ] Estructura de ambos proyectos (backend Python vs frontend React) y cómo se comunicarán
- **Ejercicios:** descargar y explorar el dataset Hard Hat Detection; montar entorno Python con OpenCV funcionando (leer y mostrar una imagen del dataset); crear el proyecto React vacío que más adelante será el dashboard.

## Módulo 1 — Fundamentos de imagen digital y NumPy aplicado
- [ ] Qué es una imagen digital: píxeles, canales (RGB/BGR), resolución, profundidad de color
- [ ] Representación de imágenes como arrays NumPy (shape, dtype)
- [ ] Operaciones básicas: recorte, redimensionado, rotación, conversión de espacios de color
- [ ] Lectura/escritura de imágenes y vídeo con OpenCV
- **Ejercicios:** cargar imágenes del dataset y extraer sus propiedades (dimensiones, canales); aplicar transformaciones básicas (recorte, redimensionado) sobre ellas; leer las anotaciones del dataset (bounding boxes de `helmet`/`head`/`person`) y dibujarlas manualmente sobre la imagen con OpenCV.

## Módulo 2 — Procesamiento clásico de imágenes
- [ ] Filtros y convoluciones (blur, sharpen, kernels personalizados)
- [ ] Detección de bordes (Canny), umbralización (thresholding)
- [ ] Operaciones morfológicas (erosión, dilatación)
- [ ] Detección de contornos y formas simples
- **Ejercicios:** pipeline de preprocesado sobre imágenes del dataset (normalización, mejora de contraste); intento de aislar cascos por color/forma sin modelos de ML, para comprobar en la práctica por qué este enfoque clásico es insuficiente y se necesita el modelo de deep learning del Módulo 4.

## Módulo 3 — Fundamentos de Deep Learning aplicado a CV
- [ ] Concepto de red neuronal convolucional (CNN): qué resuelve y por qué funciona en imágenes (sin profundizar en las matemáticas)
- [ ] Conceptos clave: entrenamiento, inferencia, dataset, overfitting (a nivel de idea, no de implementación)
- [ ] Diferencia entre clasificación, detección de objetos y segmentación
- [ ] Modelos preentrenados: qué son y por qué se usan en vez de entrenar desde cero
- **Ejercicios:** ninguno práctico (módulo conceptual) — cuestionario/discusión guiada, usando como ejemplo constante el propio caso de detección de cascos, para asentar el vocabulario antes del Módulo 4.

## Módulo 4 — Detección de cascos de seguridad con YOLO
- [ ] Instalación y uso de Ultralytics YOLO
- [ ] Inferencia sobre imágenes del dataset con un modelo preentrenado (COCO) — comprobar que detecta "person" pero no distingue casco/no casco
- [ ] Preparar el dataset Hard Hat Detection en formato YOLO (train/val, clases `helmet`/`head`/`person`)
- [ ] Fine-tuning ligero de YOLO sobre el dataset para que aprenda a distinguir casco vs no casco
- [ ] Interpretación de resultados: bounding boxes, clases, confianza (confidence score); filtrado por umbral
- **Ejercicios:** script de inferencia con el modelo preentrenado sobre imágenes del dataset; preparación del dataset en formato YOLO; entrenamiento/fine-tuning del modelo propio de detección de cascos; script final que dibuje las cajas y etiquetas (`helmet`/`no helmet`) sobre la imagen.

## Módulo 5 — Backend en Django para exponer el detector de cascos
- [ ] Fundamentos de Django: `django-admin startproject`/`startapp`, estructura de un proyecto (settings, apps, urls.py), ciclo request/response
- [ ] Modelos y ORM: `Deteccion` (imagen, fecha, nº personas con/sin casco) para guardar histórico de cada análisis en base de datos, migraciones (`makemigrations`/`migrate`)
- [ ] Panel de administración automático de Django (`/admin`) para revisar el histórico de detecciones sin construir UI propia
- [ ] Django REST Framework: serializers, views/viewsets, endpoint que recibe una imagen (upload), la pasa por el modelo del Módulo 4, guarda el resultado vía ORM y devuelve las detecciones (cajas, clase, confianza) en JSON
- [ ] Manejo de CORS (`django-cors-headers`, necesario para que React pueda consumir la API)
- **Ejercicios:** proyecto Django con app dedicada al detector; modelo `Deteccion` + migraciones; endpoint DRF `/api/detectar-cascos/` funcional usando el modelo entrenado en el Módulo 4, que además persiste cada análisis en BD; comprobación del histórico guardado desde el panel `/admin`; pruebas con Postman/curl antes de conectar con React.

## Módulo 6 — Fundamentos de React
- [ ] Componentes funcionales, JSX, props
- [ ] Estado (`useState`) y ciclo de vida (`useEffect`)
- [ ] Renderizado condicional y listas (`.map`)
- [ ] Formularios controlados, subida de archivos (`<input type="file">`)
- **Ejercicios:** componente que gestione el formulario de subida de imagen (una foto de obra) y muestre una previsualización local, como primera pieza del futuro dashboard de EPI.

## Módulo 7 — Dashboard de cumplimiento de EPI en React
- [ ] Peticiones HTTP desde React (`fetch`/`axios`), manejo de `FormData` para subir imágenes al endpoint `/api/detectar-cascos/`
- [ ] Manejo de estados de carga y error (loading, error, success)
- [ ] Renderizado de resultados dinámicos: dibujar bounding boxes sobre la imagen en el navegador (`<canvas>`), coloreando distinto casco vs no casco
- [ ] Resumen de cumplimiento (nº de personas con casco / sin casco) a partir del JSON devuelto por la API
- [ ] Organización del proyecto en componentes reutilizables
- **Ejercicios:** conectar el formulario del Módulo 6 con el endpoint `/api/detectar-cascos/` del Módulo 5; superponer las detecciones sobre la imagen; añadir el panel-resumen de cumplimiento.

## Módulo 8 — Integración final, pulido y entrega
- [ ] Manejo de errores end-to-end (backend y frontend) en el flujo completo del sistema de EPI
- [ ] Mejoras de UI/UX (estados de carga, mensajes claros, responsive básico) del dashboard
- [ ] Pruebas con un conjunto variado de imágenes del dataset (distintas condiciones: varias personas, oclusiones, ángulos)
- [ ] Documentación mínima del proyecto y preparación de demo/presentación del sistema completo
- **Ejercicios:** batería de pruebas con imágenes variadas del dataset; checklist de errores comunes revisado antes de la entrega; ensayo de la demo end-to-end (subir imagen → ver cumplimiento de EPI en el dashboard).

---

## 📌 Notas de seguimiento
- Última actualización: _pendiente de primera sesión_
- Módulo actual: _Módulo 0 — Entorno y primer contacto_

---

## 📚 Resumen de progreso por módulo

> Esta sección es el "contexto portable" del curso: si se abre un chat nuevo, leer esto (junto a los checkboxes de arriba) debería bastar para retomar exactamente donde se dejó, sin reexplicar nada al alumno.

### Módulo 0 — Entorno y primer contacto (pendiente de iniciar)
- _Sin avances registrados todavía._
