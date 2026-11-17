"""Multipaciente, fases 1-4: Takens + Rips + H1 em todas as gravações de cada paciente.

Mesmos parâmetros da análise original (FP1-F7, d=3, tau=4, janela 8 s, passo 4 s).
Um cache por paciente em resultados/cache/descritores_<paciente>_<config>.csv.
Use --refazer para recalcular caches existentes.
"""
import _caminho  # noqa: F401
import sys
import time

from tda_eeg import config as cfg
from tda_eeg import dados, extracao


def rotulo(pac):
    return f"{pac}_{extracao.rotulo_config()}"


def main():
    cfg.garantir_pastas()
    refazer = "--refazer" in sys.argv
    for pac in cfg.PACIENTES_MULTI:
        t0 = time.time()
        gs = dados.listar_gravacoes(pacientes=(pac,), restringir=None)
        n_crises = sum(len(g.crises) for g in gs)
        print(f"\n[{pac}] {len(gs)} gravações, {n_crises} crises", flush=True)
        df = extracao.extrair_ou_carregar(gs, rotulo=rotulo(pac), refazer=refazer)
        print(df.classe.value_counts().to_string())
        print(f"  {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
