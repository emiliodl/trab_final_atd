"""Baixa do PhysioNet as gravações do CHB-MIT usadas no projeto.

  chb01 (análise original, scripts 01-09): 7 EDF, cerca de 297 MB;
  chb02 e chb03 (análise multipaciente, scripts 10-12): mais 28 EDF, cerca de 1,2 GB.

Os EDF não ficam no repositório. Arquivos já presentes em dataset/ são mantidos.
Uso: python scripts/00_baixar_dados.py            (tudo)
     python scripts/00_baixar_dados.py --so-chb01 (só a análise original)
"""
import _caminho  # noqa: F401
import sys

from tda_eeg import config as cfg
from tda_eeg import dados

GRAVACOES = {
    "chb01": ["chb01_01", "chb01_02", "chb01_03", "chb01_04", "chb01_05", "chb01_06", "chb01_09"],
    # chb02: gravações 01-19 (a 16 vem em dois arquivos, 16 e 16+); contêm as 3 crises do paciente
    "chb02": [f"chb02_{i:02d}" for i in range(1, 16)] + ["chb02_16", "chb02_16+", "chb02_17",
                                                         "chb02_18", "chb02_19"],
    # chb03: gravações 01-08; contêm 4 das 7 crises do paciente
    "chb03": [f"chb03_{i:02d}" for i in range(1, 9)],
}


def main():
    pacientes = cfg.PACIENTES if "--so-chb01" in sys.argv else cfg.PACIENTES_MULTI
    for pac in pacientes:
        dados.baixar_edf(f"{pac}-summary.txt")
        for nome in GRAVACOES[pac]:
            destino = cfg.DIR_DADOS / f"{nome}.edf"
            ja_existia = destino.exists()
            dados.baixar_edf(f"{nome}.edf")
            print(f"  {nome}.edf: {'já estava na pasta' if ja_existia else 'baixado'}", flush=True)


if __name__ == "__main__":
    main()
