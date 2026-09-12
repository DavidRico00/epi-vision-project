## Módulo 3 — Fundamentos de Deep Learning aplicado a CV

### 3.1 Redes neuronales convolucionales (CNN): qué resuelven y por qué

#### El problema que resuelve una CNN

En el Módulo 2 se vio el límite del enfoque clásico: reglas de color/forma escritas a mano (HSV, contornos, circularidad) fallaban en cuanto cambiaba la luz, el color del casco o el fondo. El pipeline no tenía ninguna noción real de "esto es un casco" — solo reaccionaba a patrones de contraste local definidos manualmente.

Una CNN invierte el enfoque: en vez de que la persona escriba las reglas ("busca amarillo, luego cierra huecos, luego mira circularidad"), la red **aprende** esas reglas a partir de miles de ejemplos etiquetados (las imágenes del dataset Hard Hat Detection, con sus bounding boxes de `helmet`/`head`/`person`).

#### Por qué funciona bien en imágenes concretamente

Una CNN aplica **filtros convolucionales** — el mismo concepto de kernel que se vio en el Módulo 2 (`cv2.filter2D`) — pero con una diferencia clave: **los valores del kernel no los define la persona, los aprende la red** durante el entrenamiento, ajustándolos poco a poco para minimizar el error entre lo que predice y lo que dice la etiqueta real.

Además, una CNN apila muchas capas de estos filtros, una detrás de otra:

- Las primeras capas aprenden a detectar patrones muy simples: bordes, cambios de contraste, manchas de color (parecido conceptualmente al kernel de bordes horizontal del Módulo 2, pero aprendido, no diseñado a mano).
- Las capas intermedias combinan esos patrones simples en formas más complejas: curvas, texturas, esquinas.
- Las capas finales combinan esas formas en conceptos de alto nivel: "esto tiene la curvatura y textura típica de un casco", "esto es una cara", etc.

Es literalmente el mismo tipo de operación que ya se conoce (`filter2D` con un kernel deslizante), pero con miles de kernels distintos, organizados en capas, y aprendidos automáticamente en vez de diseñados a ojo.

#### Ejemplo aplicado al proyecto

Recordando el fallo del Módulo 2: el rango HSV `15-35` no detectaba cascos blancos ni rojos, y confundía roca/madera con el casco amarillo. Una CNN entrenada con el dataset completo (que incluye cascos amarillos, blancos, rojos, en nieve, cantera, obra con andamios) aprende automáticamente que "casco" no depende de un color fijo, sino de una combinación de forma (curvatura característica), posición relativa (sobre la cabeza de una persona) y contexto — algo que ningún kernel fijo de 5×5 podía capturar.

---

### 3.2 Entrenamiento, inferencia, dataset y overfitting

#### Entrenamiento vs inferencia

Son las dos fases separadas del ciclo de vida de un modelo:

- **Entrenamiento:** el proceso en el que la red *aprende*. Se le dan miles de imágenes con su etiqueta correcta (en este caso, las imágenes del dataset con sus bounding boxes de `helmet`/`head`/`person`), la red hace una predicción, se compara con la etiqueta real, y se ajustan los filtros internos para reducir ese error. Se repite muchas veces sobre todo el dataset (cada pasada completa se llama **época**). Es la parte lenta y computacionalmente cara.
- **Inferencia:** una vez entrenado el modelo, usarlo sobre una imagen nueva (que nunca vio) para obtener una predicción. Es rápida — es literalmente pasar la imagen "hacia adelante" por la red ya entrenada, sin ajustar nada.

En el Módulo 4 primero se hará **inferencia** con un modelo preentrenado (COCO) para comprobar sus límites, y después **fine-tuning** — que es una forma de entrenamiento, pero partiendo de un modelo ya entrenado en vez de desde cero.

#### Dataset: splits de train/validation

Un dataset no se usa entero para entrenar. Se divide típicamente en:

- **Train (entrenamiento):** las imágenes con las que la red ajusta sus filtros.
- **Validation (validación):** imágenes que la red **no ve durante el entrenamiento**, usadas para comprobar cómo generaliza a casos nuevos mientras entrena.
- A veces también hay un **test** final, separado incluso de la validación, para la evaluación definitiva.

En el Módulo 4 se preparará el dataset Hard Hat Detection exactamente en esta estructura (`train`/`val`) antes del fine-tuning.

#### Overfitting

Es el problema central a vigilar durante cualquier entrenamiento: la red "memoriza" las imágenes de entrenamiento en vez de aprender patrones generales.

Un modelo sobreajustado (*overfit*) da resultados excelentes sobre las imágenes de `train` (las que ya vio) pero falla sobre imágenes nuevas de `validation` o del mundo real — porque en vez de aprender "así se ve un casco", aprendió detalles específicos de esas fotos concretas (ese fondo exacto, esa persona exacta, ese ángulo exacto).

Se detecta comparando el error (o la precisión) en `train` frente a `validation`: si en train el modelo va muy bien pero en validation va mal, hay overfitting.

#### Ejemplo aplicado al proyecto

El pipeline clásico del Módulo 2 sirve como analogía extrema de overfitting manual: el rango HSV se "ajustó" mirando una imagen concreta (`hard_hat_workers0.png`), y ese ajuste no generalizó nada a las demás imágenes del dataset (falló con cascos rojos, blancos, con fondos de roca). Si se entrena una CNN muy pequeña con muy pocas imágenes, o durante demasiadas épocas sobre un dataset pequeño, puede pasarle algo parecido: "memoriza" los ejemplos concretos que vio en vez de aprender el concepto general de "casco".

Por eso el dataset Hard Hat Detection, con miles de imágenes en condiciones muy variadas (nieve, cantera, obra, andamios), es justo lo que permite que YOLO generalice mejor que el pipeline de color del Módulo 2.

---

### 3.3 Clasificación vs detección de objetos vs segmentación

Estas tres tareas se confunden fácilmente al principio porque todas "miran" una imagen, pero responden preguntas distintas — y es importante distinguirlas bien porque **YOLO en el Módulo 4 hace concretamente detección**, no las otras dos.

#### Clasificación (classification)

Responde a: **"¿qué hay en esta imagen?"** — una etiqueta única para toda la imagen completa, sin decir dónde está.

Ejemplo aplicado al proyecto: si se le diera a un modelo de clasificación una foto de una obra y se le preguntara "¿hay un casco en esta imagen?", respondería `sí`/`no` (o una probabilidad), pero **no diría cuántas personas hay, ni dónde está cada casco, ni si es de una persona concreta o de otra**. Ni siquiera daría una respuesta fiable a nivel de imagen completa con varias personas: normalmente entregaría una única probabilidad para toda la imagen (por ejemplo, "casco: 70%"), sin decir si ese 70% viene de 1 persona con casco y 4 sin él, o de 4 con casco y 1 sin él. La clasificación no tiene noción de "instancia individual" — por eso no sirve ni para contar ni para localizar. Para el sistema de EPI esto es insuficiente — se necesita saber *cuántas* personas cumplen y cuáles no, individualmente.

#### Detección de objetos (object detection) — esto es lo que necesita el proyecto

Responde a: **"¿qué hay, y dónde exactamente?"** — para cada objeto de interés en la imagen, devuelve una etiqueta (`helmet`, `head`, `person`) **más** un bounding box (las coordenadas del rectángulo que lo encierra) **más** una confianza (qué tan seguro está el modelo).

Es exactamente el formato de las anotaciones ya trabajadas en el Módulo 1 (`ej4_leer_xml.py`, que parseaba `xmin`/`ymin`/`xmax`/`ymax` de cada objeto) — YOLO en el Módulo 4 va a producir esa misma estructura de salida, pero calculada por el modelo en vez de leída de un XML preexistente.

Ejemplo aplicado al proyecto: sobre una foto con 4 trabajadores, un detector devuelve algo como 4 cajas etiquetadas `person`, más las cajas correspondientes de `helmet`/`head` superpuestas sobre cada cabeza — justo lo que se necesita para contar cumplimiento por persona.

#### Segmentación (segmentation)

Responde a: **"¿qué píxeles exactos pertenecen a cada objeto?"** — en vez de un rectángulo aproximado, produce una máscara que sigue el contorno exacto del objeto, píxel a píxel.

Ejemplo aplicado al proyecto: la segmentación diría exactamente qué píxeles son "casco" (incluyendo su forma curva real) frente a "no casco", en vez de un simple rectángulo que también incluye algo de fondo alrededor. Es más precisa, pero también más cara de calcular y de anotar (etiquetar máscaras píxel a píxel es mucho más costoso que dibujar un rectángulo) — y para este caso de uso (contar cumplimiento, no medir la forma exacta del casco) es una precisión que no se necesita.

#### Por qué detección es la elección correcta aquí

Para "¿cuántas personas llevan casco y cuántas no?", un bounding box por persona/casco es toda la información necesaria — no hace falta saber el contorno exacto del casco píxel a píxel. Por eso el proyecto usa YOLO (un detector), no un modelo de segmentación, aunque también existen variantes de YOLO que segmentan si en algún momento se quisiera ese nivel de detalle.

---

### 3.4 Modelos preentrenados: qué son y por qué se usan

#### Qué es un modelo preentrenado

Es un modelo que **ya fue entrenado por otros**, sobre un dataset grande y genérico, antes de tocarlo. En el caso de YOLO, la versión que se usará en el Módulo 4 viene preentrenada sobre **COCO** (Common Objects in Context) — un dataset público con más de 80 clases genéricas: `person`, `car`, `dog`, `chair`, `bottle`, etc. — pero **no** tiene las clases específicas necesarias (`helmet`, `head`).

#### Por qué no se entrena desde cero

Entrenar una CNN de detección desde cero (con los pesos internos empezando en valores aleatorios) requiere:

- **Muchísimos datos** — normalmente cientos de miles o millones de imágenes — para que la red aprenda desde "no distingue nada" hasta reconocer patrones útiles.
- **Muchísimo cómputo** — días o semanas incluso con hardware potente.

El dataset Hard Hat Detection tiene 5000 imágenes — suficiente para *ajustar* un modelo ya capaz, pero muy poco para entrenar uno desde cero sin caer en overfitting severo (ver 3.2): con tan pocos datos, una red que arranca de cero memorizaría esas 5000 imágenes en vez de aprender el concepto general de "casco".

#### Transfer learning: la solución intermedia

La idea clave (y es lo que se hará literalmente en el Módulo 4) es el **transfer learning** ("aprendizaje por transferencia"): se parte de un modelo que **ya sabe ver** — ya aprendió, entrenando sobre COCO, a detectar bordes, texturas, formas, y el concepto general de "persona" — y solo se le enseña la parte específica que le falta: distinguir `helmet` de `head` sobre cabezas de personas.

Esto es **fine-tuning**: se sigue entrenando el modelo preentrenado, pero partiendo de esos pesos ya útiles en vez de valores aleatorios, y normalmente con muchas menos imágenes y mucho menos tiempo de cómputo que entrenar desde cero.

#### Ejemplo aplicado al proyecto (esto es literalmente el plan del Módulo 4)

1. Se carga YOLO preentrenado en COCO → se prueba sobre imágenes del dataset → detecta bien `person`, porque COCO ya tiene esa clase, pero no distingue casco/no casco, porque COCO nunca vio esas etiquetas.
2. Se hace fine-tuning: se sigue entrenando ese mismo modelo, pero ahora con el dataset Hard Hat Detection (clases `helmet`/`head`/`person`), aprovechando que ya "sabe ver" formas y texturas en general.
3. El resultado es un modelo que detecta bien las 3 clases, entrenado en un tiempo razonable y con solo 5000 imágenes — algo imposible si se hubiera entrenado desde cero.
