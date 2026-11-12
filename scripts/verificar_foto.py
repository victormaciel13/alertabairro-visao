"""Verifica se a foto de um relato mostra alagamento.

É a mesma lógica que a API do AlertaBairro usa para marcar um relato como
"verificado por imagem".

Uso:
    python scripts/verificar_foto.py --foto minha_foto.jpg
"""
import argparse
import json

from ultralytics import YOLO


def verificar(caminho_foto, modelo="modelo/alertabairro_v1.pt", conf=0.5):
    r = YOLO(modelo).predict(caminho_foto, conf=conf, verbose=False)[0]
    confs = [round(float(c), 3) for c in r.boxes.conf]
    return {
        "alagamento_confirmado": len(confs) > 0,
        "regioes_detectadas": len(confs),
        "confianca_maxima": max(confs) if confs else 0.0,
        "limiar": conf,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--foto", required=True)
    p.add_argument("--modelo", default="modelo/alertabairro_v1.pt")
    p.add_argument("--conf", type=float, default=0.5)
    a = p.parse_args()
    print(json.dumps(verificar(a.foto, a.modelo, a.conf), ensure_ascii=False, indent=2))
