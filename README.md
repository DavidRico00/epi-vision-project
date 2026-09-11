# 🦺 EPI Vision — Sistema de monitorización de cascos de seguridad

Sistema de visión artificial que detecta personas en imágenes de obra/planta y determina si llevan puesto el casco de seguridad (EPI), mostrando el resultado en un dashboard web con las detecciones marcadas y un resumen de cumplimiento.

Proyecto desarrollado como parte de un plan de estudio intensivo de **Computer Vision + React**, combinando procesamiento de imagen clásico, detección de objetos con deep learning (YOLO), un backend que expone el modelo como API (Django + DRF), y un frontend en React que consume esa API.

## 📸 Qué hace

A partir de una imagen o vídeo de una obra, el sistema:
1. Detecta a las personas presentes en la escena.
2. Determina si cada una lleva puesto el casco de seguridad.
3. Muestra las detecciones superpuestas sobre la imagen en un dashboard web.
4. Genera un resumen de cumplimiento (nº de personas con/sin casco).
5. Guarda un histórico de cada análisis realizado.

## 🗂️ Dataset

[Hard Hat Detection (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection) — 5000 imágenes anotadas en formato Pascal VOC (XML), con las clases `helmet`, `head` (sin casco) y `person`.

## 🏗️ Estructura del repositorio

```
epi-vision-project/
├── backend/                # Python — procesamiento de imagen, modelo YOLO y API
│   ├── pyproject.toml       # dependencias gestionadas con uv
│   ├── .python-version
│   ├── scripts/              # scripts de exploración y prueba
│   └── data/                 # dataset (no versionado, ver instrucciones abajo)
├── frontend/                # React (Vite) — dashboard de cumplimiento de EPI
│   ├── package.json
│   └── src/
├── docs/
│   ├── plan-estudio-vision-artificial-react.md   # plan de estudio y progreso módulo a módulo
└── README.md
```

## 🛠️ Stack técnico

**Backend:** Python · [uv](https://docs.astral.sh/uv/) (gestión de entorno y dependencias) · OpenCV · NumPy · Ultralytics (YOLO) · Django · Django REST Framework

**Frontend:** React · Vite · ESLint

## 🚀 Puesta en marcha

### Backend

```bash
cd backend
uv sync
```

Descarga del dataset (requiere cuenta y token de API de [Kaggle](https://www.kaggle.com)):

```bash
uv run kaggle datasets download -d andrewmvd/hard-hat-detection -p data --unzip
```

Ejecutar un script:

```bash
uv run python scripts/nombre_del_script.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Servidor de desarrollo disponible en `http://localhost:5173`.

## 📄 Licencia

Proyecto con fines educativos. Dataset bajo licencia propia de Kaggle/[CC BY 4.0](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection).