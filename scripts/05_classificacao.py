"""Tópico avançado - Machine Learning: detecção de janelas ictais.

Entrada: descritores topológicos de H1 por janela. Comparação com baselines
clássicos de EEG. Validação deixa-uma-gravação-fora (ver tda_eeg/avaliacao.py).

Positivas: janelas ictais. Negativas: interictais e pré-ictais. Janelas de
transição e pós-ictais são excluídas do treino e das métricas.
"""
import _caminho  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from tda_eeg import avaliacao, baseline, estilo, extracao, topologia
from tda_eeg import config as cfg
from tda_eeg import dados

CORES = [estilo.AZUL, estilo.LARANJA, estilo.AGUA, estilo.AMARELO, estilo.VIOLETA]


def conjuntos(df):
    alt = [c + "_alt" for c in topologia.COLUNAS_TOPO]
    c = {"Topológico H1 (d=3, τ=4)": topologia.COLUNAS_TOPO}
    if alt[0] in df:
        c["Topológico H1 (d, τ pelos dados)"] = alt
    c["Clássico: só forma"] = baseline.COLUNAS_FORMA
    c["Clássico: forma + amplitude"] = baseline.COLUNAS_BASE
    c["Topológico + clássico"] = topologia.COLUNAS_TOPO + baseline.COLUNAS_BASE
    return c


def carregar():
    import json
    df = extracao.carregar(extracao.rotulo_config())
    arq = cfg.DIR_TAB / "parametros_takens.json"
    if arq.exists():
        p = json.loads(arq.read_text(encoding="utf-8"))
        rot = extracao.rotulo_config(dim=int(p["dim_sugerida"]), atraso=int(p["atraso_sugerido"]))
        if (cfg.DIR_CACHE / f"descritores_{rot}.csv").exists():
            alt = extracao.carregar(rot)[["gravacao", "t_ini"] + topologia.COLUNAS_TOPO]
            alt = alt.rename(columns={c: c + "_alt" for c in topologia.COLUNAS_TOPO})
            df = df.merge(alt, on=["gravacao", "t_ini"], how="left", validate="one_to_one")
    return df


def metricas_evento(df, escores, y, validas):
    """Alarmes falsos por hora no limiar que detecta 1 ou as 2 crises."""
    neg = validas & (y == 0)
    horas = neg.sum() * cfg.PASSO_S / 3600
    picos = [escores[(df.gravacao == g).to_numpy() & (y == 1)].max()
             for g in df.loc[y == 1, "gravacao"].unique()]

    def falsos(limiar):
        return sum(avaliacao.sequencias((escores >= limiar)[m] & neg[m])
                   for m in [(df.gravacao == g).to_numpy() for g in df.gravacao.unique()])

    return {"fa_h_1_crise": falsos(max(picos)) / horas, "fa_h_2_crises": falsos(min(picos)) / horas}


def main():
    cfg.garantir_pastas()
    estilo.aplicar()
    gravs = dados.listar_gravacoes()
    df = carregar()
    validas = df.classe.isin(["ictal", "interictal", "pre-ictal"]).to_numpy()
    y = (df.classe == "ictal").to_numpy().astype(int)
    com_crise = [g.nome for g in gravs if g.crises]

    linhas, guardados = [], {}
    for nome_m, modelo in avaliacao.modelos().items():
        for nome_c, cols in conjuntos(df).items():
            s = avaliacao.pontuar_fora_da_amostra(df, cols, y, validas, modelo)
            guardados[(nome_m, nome_c)] = s
            r = {"modelo": nome_m, "descritores": nome_c, "n_descritores": len(cols),
                 "auroc": roc_auc_score(y[validas], s[validas]),
                 "auprc": average_precision_score(y[validas], s[validas])}
            for g in com_crise:
                m = validas & (df.gravacao == g).to_numpy()
                r[f"auroc_{g}"] = roc_auc_score(y[m], s[m])
            r.update(metricas_evento(df, s, y, validas))
            linhas.append(r)
            print(f"  {nome_m:20s} {nome_c:34s} AUROC {r['auroc']:.3f}")
    tab = pd.DataFrame(linhas)
    tab.round(4).to_csv(cfg.DIR_TAB / "classificacao_deteccao.csv", index=False)
    prev = y[validas].mean()
    print(f"\nJanelas: {validas.sum()} ({y[validas].sum()} ictais, prevalência {prev:.4f})")
    pd.set_option("display.width", 220)
    print(tab.round(3).to_string(index=False))

    # --- figura 1: ROC e AUROC por crise --------------------------------------
    mod = "Regressão logística"
    nomes = list(conjuntos(df))
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True,
                               gridspec_kw={"width_ratios": [1, 1.15]})
    a.plot([0, 1], [0, 1], color=estilo.EIXO, lw=1)
    for nome, cor in zip(nomes, CORES):
        s = guardados[(mod, nome)]
        fpr, tpr, _ = roc_curve(y[validas], s[validas])
        auc = tab[(tab.modelo == mod) & (tab.descritores == nome)].auroc.item()
        a.plot(fpr, tpr, color=cor, lw=2, label=f"{nome} — {auc:.2f}")
    a.set_xlabel("taxa de falsos positivos")
    a.set_ylabel("sensibilidade")
    a.set_aspect("equal")
    a.set_title(f"Curva ROC fora da amostra ({mod.lower()})")
    a.legend(loc="lower right", title="descritores — AUROC", title_fontsize=8.5, fontsize=8.5)

    yy = np.arange(len(nomes))[::-1]
    b.axvline(0.5, color=estilo.EIXO, lw=1.2)
    for g, marc in zip(com_crise, ["o", "D"]):
        v = [tab[(tab.modelo == mod) & (tab.descritores == n)][f"auroc_{g}"].item() for n in nomes]
        b.scatter(v, yy, s=70, marker=marc, c=CORES[: len(nomes)], zorder=3,
                  edgecolor=estilo.SUPERFICIE, linewidths=1)
        b.scatter([], [], s=60, marker=marc, color=estilo.TINTA_2, label=f"crise de {g}")
        for vi, yi in zip(v, yy):
            b.annotate(f"{vi:.2f}", (vi, yi), textcoords="offset points",
                       xytext=(0, 9 if marc == "o" else -15), ha="center", fontsize=8.5,
                       color=estilo.TINTA_2)
    b.set_yticks(yy, nomes)
    b.set_ylim(-0.7, len(nomes) - 0.3)
    b.set_xlim(0.3, 1.03)
    b.grid(axis="y", visible=False)
    b.set_xlabel("AUROC na gravação retida (0,5 = acaso)")
    b.set_title("AUROC em cada crise, com a gravação inteira fora do treino")
    b.legend(loc="lower left")
    fig.suptitle("Detecção de janelas ictais: descritores topológicos × baselines clássicos")
    estilo.salvar(fig, cfg.DIR_FIG / "05_classificacao_deteccao.png")

    # --- figura 2: escore ao longo do tempo em torno de cada crise ------------
    pares = [("Topológico H1 (d=3, τ=4)", estilo.AZUL), ("Clássico: forma + amplitude", estilo.AGUA)]
    fig, eixos = plt.subplots(2, len(com_crise), figsize=(12.5, 5.6), sharex=True, sharey=True,
                              constrained_layout=True)
    for j, g in enumerate(gravs_com := [g for g in gravs if g.crises]):
        ini, fim = g.crises[0]
        m = (df.gravacao == g.nome).to_numpy()
        t = (df.t_ini[m] + cfg.JANELA_S / 2 - ini).to_numpy()
        perto = (t > -600) & (t < 400)
        for i, (nome, cor) in enumerate(pares):
            ax = eixos[i, j]
            estilo.faixa_crise(ax, 0, fim - ini, "crise" if i == 0 else None)
            ax.plot(t[perto], guardados[(mod, nome)][m][perto], color=cor, lw=1.6)
            ax.set_title(f"{nome} — {g.nome}")
            ax.set_ylim(-0.03, 1.03)
    for ax in eixos[:, 0]:
        ax.set_ylabel("escore de crise")
    for ax in eixos[-1]:
        ax.set_xlabel("tempo relativo ao início da crise (s)")
    fig.suptitle("Escore do classificador ao longo do tempo (modelo treinado sem a gravação mostrada)")
    estilo.salvar(fig, cfg.DIR_FIG / "05_escores_em_torno_das_crises.png")


if __name__ == "__main__":
    main()
