"""Fases 1-4 em todas as gravações: EDF -> filtro -> janelas -> Takens -> Rips -> H1.

Gera, em resultados/cache:
  * descritores do canal FP1-F7 com os parâmetros do enunciado (janela 8 s, passo 4 s);
  * um recorte ao redor de cada crise com passo de 1 s (para as figuras de detalhe);
  * a mesma extração com (d, tau) escolhidos pelos critérios do script 02, se existir.
"""
import _caminho  # noqa: F401
import json
import time

from tda_eeg import config as cfg
from tda_eeg import dados, extracao

MARGEM_ANTES, MARGEM_DEPOIS = 300.0, 180.0


def main():
    cfg.garantir_pastas()
    gravs = dados.listar_gravacoes()
    print("Gravações:", ", ".join(g.nome for g in gravs))
    for g in gravs:
        for ini, fim in g.crises:
            print(f"  crise em {g.nome}: {ini:.0f}-{fim:.0f} s")

    t0 = time.time()
    print("\n[1] Configuração do enunciado (d=3, tau=4, janela 8 s, passo 4 s)")
    df = extracao.extrair_ou_carregar(gravs, refazer=True)
    print(df.classe.value_counts().to_string())
    print(f"  classes censuradas: {int(df.h1_censuradas.sum())} | {time.time() - t0:.0f} s")

    print("\n[2] Recorte peri-ictal com passo de 1 s")
    recorte = {g.nome: (max(0.0, g.crises[0][0] - MARGEM_ANTES),
                        min(g.duracao, g.crises[0][1] + MARGEM_DEPOIS))
               for g in gravs if g.crises}
    extracao.extrair_ou_carregar(gravs, rotulo="zoom", refazer=True, passo_s=1.0, recorte=recorte)

    arq = cfg.DIR_TAB / "parametros_takens.json"
    if arq.exists():
        p = json.loads(arq.read_text(encoding="utf-8"))
        d, tau = int(p["dim_sugerida"]), int(p["atraso_sugerido"])
        print(f"\n[3] Configuração orientada pelos dados (d={d}, tau={tau})")
        extracao.extrair_ou_carregar(gravs, refazer=True, dim=d, atraso=tau)
    print(f"\nTempo total: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
