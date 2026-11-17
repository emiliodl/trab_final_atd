"""Baixa do PhysioNet as gravações do CHB-MIT usadas no projeto.

  estudo de caso (scripts 01-09): 7 EDF do chb01, cerca de 297 MB;
  análise multipaciente (scripts 10-14): o chb01 inteiro (42 EDF), o chb02 (20), o
  chb03 (8), o chb04 (5) e uma gravação de chb06, chb08 e chb12; cerca de 3,7 GB no total.

Os EDF não ficam no repositório. Arquivos já presentes em dataset/ são mantidos.
Uso: python scripts/00_baixar_dados.py            (tudo)
     python scripts/00_baixar_dados.py --so-chb01 (só o estudo de caso)
"""
import _caminho  # noqa: F401
import sys

from tda_eeg import config as cfg
from tda_eeg import dados

GRAVACOES = {
    # chb01: todas as 42 gravações do resumo (não existem 28, 35, 44 e 45); contêm as 7 crises
    "chb01": [f"chb01_{i:02d}" for i in list(range(1, 28)) + list(range(29, 35)) + list(range(36, 44)) + [46]],
    # chb02: gravações 01-19 (a 16 vem em dois arquivos, 16 e 16+); contêm as 3 crises do paciente
    "chb02": [f"chb02_{i:02d}" for i in range(1, 16)] + ["chb02_16", "chb02_16+", "chb02_17",
                                                         "chb02_18", "chb02_19"],
    # chb03: gravações 01-08; contêm 4 das 7 crises do paciente
    "chb03": [f"chb03_{i:02d}" for i in range(1, 9)],
    # chb04: gravações 01-05 (4 h cada); 1 das 4 crises. chb06, chb08 e chb12: uma gravação cada.
    "chb04": [f"chb04_{i:02d}" for i in range(1, 6)],
    "chb06": ["chb06_01"],
    "chb08": ["chb08_02"],
    "chb12": ["chb12_06"],
}


def main():
    so_estudo = "--so-chb01" in sys.argv
    for pac in (cfg.PACIENTES if so_estudo else cfg.PACIENTES_MULTI):
        dados.baixar_edf(f"{pac}-summary.txt")
        for nome in (cfg.GRAVACOES_ESTUDO[pac] if so_estudo else GRAVACOES[pac]):
            destino = cfg.DIR_DADOS / f"{nome}.edf"
            ja_existia = destino.exists()
            dados.baixar_edf(f"{nome}.edf")
            print(f"  {nome}.edf: {'já estava na pasta' if ja_existia else 'baixado'}", flush=True)


if __name__ == "__main__":
    main()
