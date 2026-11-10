"""Sensibilidade (exploratória): os resultados dependem do canal e de (d, tau)?

Para cada configuração, mede a separação ictal x interictal (AUROC univariado) dos
descritores de H1. Nenhum modelo é ajustado. ATENÇÃO: escolher o "melhor" canal ou
parâmetro a partir destas tabelas e depois reportar seu desempenho seria seleção
com os mesmos dados; por isso a análise principal usa os valores fixados a priori.

Usa o motor Ripser (mesma persistência do GUDHI, muito mais rápido); a
equivalência numérica é conferida aqui em uma amostra de janelas.
"""
import _caminho  # noqa: F401
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tda_eeg import config as cfg
from tda_eeg import dados, estilo, extracao, takens, topologia

DIMS = [2, 3, 4, 5]
ATRASOS = [2, 4, 7, 10, 14]
DESCR = ["h1_pers_max", "h1_pers_total", "h1_entropia", "h1_pais2_l1"]
NOMES = {"h1_pers_max": "persistência máxima", "h1_pers_total": "persistência total",
         "h1_entropia": "entropia", "h1_pais2_l1": "paisagem λ2 (L1)",
         "log_variancia": "log-variância (baseline)"}


def auc(df, col, mesma_gravacao):
    pos = df.loc[df.classe == "ictal", col].to_numpy(float)
    neg = df[df.classe == "interictal"]
    if mesma_gravacao:
        neg = neg[neg.gravacao.isin(df.loc[df.classe == "ictal", "gravacao"].unique())]
    neg = neg[col].to_numpy(float)
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])


def linha(df, extras, colunas):
    d = dict(extras)
    for c in colunas:
        d[f"{c}__todas"] = auc(df, c, False)
        d[f"{c}__mesma_grav"] = auc(df, c, True)
    return d


def conferir_motores(gravs, n=60):
    rng = np.random.default_rng(cfg.SEMENTE)
    x, fs = dados.carregar_canal(gravs[0].caminho)
    w = int(cfg.JANELA_S * fs)
    pior = 0.0
    for i in rng.integers(0, len(x) - w, n):
        X = takens.imersao_takens(x[i: i + w], cfg.DIM, cfg.ATRASO, cfg.PASSO_NUVEM)
        a = np.sort(topologia.persistencias(topologia.h1_rips(X, motor="gudhi")[0]))[::-1]
        b = np.sort(topologia.persistencias(topologia.h1_rips(X, motor="ripser")[0]))[::-1]
        a, b = a[a > 1e-6], b[b > 1e-6]
        if len(a) != len(b):
            return {"n_janelas": n, "equivalente": False, "maior_diferenca": None}
        pior = max(pior, float(np.abs(a - b).max()) if len(a) else 0.0)
    return {"n_janelas": n, "equivalente": pior < 1e-5, "maior_diferenca": pior}


def mapa(ax, M, titulo, linhas, colunas):
    im = ax.imshow(M, cmap=estilo.DIV, vmin=0.2, vmax=0.8, aspect="auto")
    ax.set_xticks(range(len(colunas)), colunas)
    ax.set_yticks(range(len(linhas)), linhas)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color=estilo.TINTA if abs(M[i, j] - 0.5) < 0.2 else "#ffffff")
    ax.set_title(titulo)
    return im


def main():
    cfg.garantir_pastas()
    estilo.aplicar()
    gravs = dados.listar_gravacoes()
    comum = dict(com_bottleneck=False, motor="ripser", verbose=False)

    conf = conferir_motores(gravs)
    print("GUDHI x Ripser:", conf)

    # --- (d, tau) no canal do enunciado --------------------------------------
    t0 = time.time()
    linhas = []
    for d in DIMS:
        for tau in ATRASOS:
            df, _ = extracao.extrair(gravs, dim=d, atraso=tau, com_baseline=False, **comum)
            linhas.append(linha(df, {"dim": d, "atraso": tau, "n_pontos": int(df.n_pontos.iloc[0])}, DESCR))
            print(f"  d={d} tau={tau:2d} | {time.time() - t0:.0f} s")
    grade = pd.DataFrame(linhas)
    grade.round(4).to_csv(cfg.DIR_TAB / "sensibilidade_d_tau.csv", index=False)

    # --- canais, com (d, tau) do enunciado ------------------------------------
    canais = dados.nomes_canais(gravs[0].caminho)
    linhas = []
    for c in canais:
        df, _ = extracao.extrair(gravs, canal=c, com_baseline=True, **comum)
        linhas.append(linha(df, {"canal": c}, DESCR + ["log_variancia"]))
        print(f"  canal {c} | {time.time() - t0:.0f} s")
    por_canal = pd.DataFrame(linhas)
    por_canal.round(4).to_csv(cfg.DIR_TAB / "sensibilidade_canais.csv", index=False)

    (cfg.DIR_TAB / "sensibilidade_resumo.json").write_text(json.dumps({
        "motores": conf, "dims": DIMS, "atrasos": ATRASOS, "canais": canais,
        "referencia": "interictal das mesmas gravacoes (chb01_03, chb01_04)",
    }, indent=2), encoding="utf-8")

    # --- figura 1: mapas d x tau ----------------------------------------------
    fig, eixos = plt.subplots(1, 4, figsize=(14, 3.6), constrained_layout=True)
    for ax, c in zip(eixos, DESCR):
        M = grade.pivot(index="dim", columns="atraso", values=f"{c}__mesma_grav").to_numpy()
        im = mapa(ax, M, NOMES[c], [f"d={d}" for d in DIMS], [str(a) for a in ATRASOS])
        ax.set_xlabel("atraso τ (amostras a 64 Hz)")
    cb = fig.colorbar(im, ax=eixos, shrink=0.85, pad=0.01)
    cb.set_label("AUROC ictal × interictal")
    cb.outline.set_visible(False)
    fig.suptitle(f"Sensibilidade a (d, τ) em {cfg.CANAL}: AUROC ictal × interictal das mesmas "
                 "gravações (0,5 = sem separação)")
    estilo.salvar(fig, cfg.DIR_FIG / "08_sensibilidade_d_tau.png")

    # --- figura 2: canais ------------------------------------------------------
    series = [("log_variancia", estilo.MUDO, "s"), ("h1_pers_max", estilo.AZUL, "o"),
              ("h1_pais2_l1", estilo.LARANJA, "D")]
    fig, ax = plt.subplots(figsize=(8.2, 7.6), constrained_layout=True)
    y = np.arange(len(canais))[::-1]
    ax.axvline(0.5, color=estilo.EIXO, lw=1.2)
    for c, cor, marc in series:
        ax.scatter(por_canal[f"{c}__mesma_grav"], y, s=44, color=cor, marker=marc, zorder=3,
                   edgecolor=estilo.SUPERFICIE, linewidths=0.8, label=NOMES[c])
    ax.set_yticks(y, canais)
    ax.set_xlim(0, 1.02)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("AUROC ictal × interictal das mesmas gravações (0,5 = sem separação)")
    ax.set_title("Separação ictal × interictal por canal (d=3, τ=4)")
    ax.legend(loc="lower left", ncols=1)
    estilo.salvar(fig, cfg.DIR_FIG / "08_sensibilidade_canais.png")
    print(grade.round(3).to_string(index=False))
    print(por_canal.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
