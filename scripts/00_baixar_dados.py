"""Baixa do PhysioNet as gravações do CHB-MIT usadas no projeto (cerca de 297 MB).

Os EDF não ficam no repositório. Arquivos já presentes em dataset/ são mantidos.
Uso: python scripts/00_baixar_dados.py
"""
import _caminho  # noqa: F401

from tda_eeg import config as cfg
from tda_eeg import dados

GRAVACOES = ["chb01_01", "chb01_02", "chb01_03", "chb01_04", "chb01_05", "chb01_06", "chb01_09"]


def main():
    for nome in GRAVACOES:
        destino = cfg.DIR_DADOS / f"{nome}.edf"
        ja_existia = destino.exists()
        dados.baixar_edf(f"{nome}.edf")
        print(f"  {nome}.edf: {'já estava na pasta' if ja_existia else 'baixado'}")


if __name__ == "__main__":
    main()
