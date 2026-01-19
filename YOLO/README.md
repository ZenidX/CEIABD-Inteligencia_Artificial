# YOLO Video Detection

Aplicación de detección de objetos en video usando YOLO (v8-v11) y la librería supervision.

## Características

- Detección de objetos en tiempo real (webcam) o en archivos de video
- Tracking de objetos con ByteTrack
- Anotaciones visuales: bounding boxes, etiquetas, trazas de movimiento
- Soporte para múltiples modelos YOLO (v8, v9, v10, v11)

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Webcam
python video_detector.py

# Archivo de video
python video_detector.py --source mi_video.mp4

# Guardar resultado
python video_detector.py --source mi_video.mp4 --output resultado.mp4

# Usar modelo diferente
python video_detector.py --model yolov8s.pt --confidence 0.6
```

Presiona `q` para salir durante la visualización.

## Modelos Disponibles

| Modelo | Velocidad | Precisión |
|--------|-----------|-----------|
| `yolo11n.pt` | Muy rápido | Básica |
| `yolo11s.pt` | Rápido | Buena |
| `yolo11m.pt` | Medio | Mejor |
| `yolo11l.pt` | Lento | Alta |
| `yolo11x.pt` | Muy lento | Máxima |

También disponibles: `yolov8*.pt`, `yolov9*.pt`, `yolov10*.pt`

## Referencias

- [Ultralytics YOLO](https://docs.ultralytics.com/models/)
- [Supervision Library](https://supervision.roboflow.com/)
