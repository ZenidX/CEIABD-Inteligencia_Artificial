"""
YOLO Video Object Detection with Supervision
=============================================
Detecta objetos en video usando modelos YOLO (v8-v11) y la librería supervision
para anotaciones y tracking.

Uso:
    python video_detector.py --source video.mp4
    python video_detector.py --source 0  # webcam
    python video_detector.py --source video.mp4 --model yolov8n.pt --output resultado.mp4
"""

import argparse
import numpy as np
import supervision as sv
from ultralytics import YOLO


def create_annotators(track: bool = True) -> dict:
    """Crea los anotadores de supervision para visualización."""
    annotators = {
        "box": sv.BoxAnnotator(thickness=2),
        "label": sv.LabelAnnotator(text_thickness=1, text_scale=0.5),
    }
    if track:
        annotators["trace"] = sv.TraceAnnotator(thickness=2, trace_length=50)
    return annotators


def process_frame(
    frame: np.ndarray,
    model: YOLO,
    tracker: sv.ByteTrack | None,
    annotators: dict,
    confidence: float,
) -> np.ndarray:
    """Procesa un frame: detección, tracking y anotación."""
    # Inferencia YOLO
    results = model(frame, conf=confidence, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Aplicar tracking si está habilitado
    if tracker is not None:
        detections = tracker.update_with_detections(detections)

    # Generar etiquetas
    if tracker is not None and detections.tracker_id is not None:
        labels = [
            f"#{tid} {model.names[cid]} {conf:.2f}"
            for tid, cid, conf in zip(
                detections.tracker_id,
                detections.class_id,
                detections.confidence,
            )
        ]
    else:
        labels = [
            f"{model.names[cid]} {conf:.2f}"
            for cid, conf in zip(detections.class_id, detections.confidence)
        ]

    # Anotar frame
    annotated = frame.copy()
    annotated = annotators["box"].annotate(annotated, detections=detections)
    annotated = annotators["label"].annotate(
        annotated, detections=detections, labels=labels
    )
    if "trace" in annotators and tracker is not None:
        annotated = annotators["trace"].annotate(annotated, detections=detections)

    return annotated


def run_detection(
    source: str,
    model_path: str = "yolo11n.pt",
    output_path: str | None = None,
    confidence: float = 0.5,
    track: bool = True,
    show: bool = True,
) -> None:
    """
    Ejecuta detección de objetos en video.

    Args:
        source: Ruta al video o índice de webcam (ej: "0" para webcam)
        model_path: Modelo YOLO a usar (yolov8n.pt, yolo11n.pt, etc.)
        output_path: Ruta para guardar el video procesado (opcional)
        confidence: Umbral de confianza para detecciones
        track: Habilitar tracking de objetos
        show: Mostrar video en tiempo real
    """
    print(f"Cargando modelo: {model_path}")
    model = YOLO(model_path)

    # Configurar tracker y anotadores
    tracker = sv.ByteTrack() if track else None
    annotators = create_annotators(track=track)

    # Determinar si es webcam o archivo
    try:
        source_idx = int(source)
        is_webcam = True
    except ValueError:
        source_idx = source
        is_webcam = False

    if is_webcam:
        # Modo webcam con OpenCV
        import cv2

        cap = cv2.VideoCapture(source_idx)
        if not cap.isOpened():
            print(f"Error: No se puede abrir la webcam {source_idx}")
            return

        print("Presiona 'q' para salir")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated = process_frame(frame, model, tracker, annotators, confidence)

            if show:
                cv2.imshow("YOLO Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()

    else:
        # Modo archivo de video con supervision
        video_info = sv.VideoInfo.from_video_path(source)
        print(f"Video: {video_info.width}x{video_info.height} @ {video_info.fps} FPS")
        print(f"Total frames: {video_info.total_frames}")

        def callback(frame: np.ndarray, frame_idx: int) -> np.ndarray:
            if frame_idx % 100 == 0:
                print(f"Procesando frame {frame_idx}/{video_info.total_frames}")
            return process_frame(frame, model, tracker, annotators, confidence)

        if output_path:
            print(f"Guardando resultado en: {output_path}")
            sv.process_video(
                source_path=source,
                target_path=output_path,
                callback=callback,
            )
            print("Video guardado correctamente")

        if show:
            import cv2

            for frame in sv.get_video_frames_generator(source):
                annotated = process_frame(frame, model, tracker, annotators, confidence)
                cv2.imshow("YOLO Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Detección de objetos en video con YOLO y supervision"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Ruta al video o índice de webcam (default: 0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Modelo YOLO (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta para guardar el video procesado",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Umbral de confianza (default: 0.5)",
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Deshabilitar tracking de objetos",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="No mostrar video en tiempo real",
    )

    args = parser.parse_args()

    run_detection(
        source=args.source,
        model_path=args.model,
        output_path=args.output,
        confidence=args.confidence,
        track=not args.no_track,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
