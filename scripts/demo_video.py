"""Demo em tempo real: roda o detector de alagamento em um vídeo e grava o resultado.

Uso:
    python scripts/demo_video.py --video rua_alagada.mp4 --saida rua_alagada_detectado.mp4
"""
import argparse
import time

import cv2
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="vídeo de entrada")
    p.add_argument("--saida", default="rua_alagada_detectado.mp4", help="vídeo de saída")
    p.add_argument("--modelo", default="modelo/alertabairro_v1.pt")
    p.add_argument("--conf", type=float, default=0.5, help="confiança mínima")
    a = p.parse_args()

    model = YOLO(a.modelo)
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"Não foi possível abrir o vídeo: {a.video}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(a.saida, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    frames, com_agua, t0 = 0, 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, conf=a.conf, verbose=False)[0]
        img = r.plot()
        alagado = len(r.boxes) > 0
        com_agua += alagado
        frames += 1
        fps = frames / (time.time() - t0)
        texto = "ALAGAMENTO DETECTADO" if alagado else "Sem alagamento"
        cor = (0, 0, 255) if alagado else (0, 200, 0)
        cv2.putText(img, f"{texto} | {fps:.0f} FPS", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
        out.write(img)

    cap.release()
    out.release()
    if frames:
        print(f"{frames} quadros | {frames / (time.time() - t0):.1f} FPS médio | "
              f"alagamento em {com_agua / frames:.0%} dos quadros")
    print("Vídeo salvo em:", a.saida)


if __name__ == "__main__":
    main()
