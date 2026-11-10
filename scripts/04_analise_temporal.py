"""Fase 5: descritores de H1 ao longo do tempo, comparados às anotações de crise.

Também cobre o tópico avançado "Estabilidade e transições": distância de
bottleneck d_B(D_t, D_{t-1}) entre janelas consecutivas.

Saídas: figuras 04_*.png e tabelas estatisticas_classes.csv / teste_blocos.csv.
As comparações são EXPLORATÓRIAS: há apenas duas crises de um paciente e as
janelas vizinhas são fortemente correlacionadas.
"""
import _caminho  # noqa: F401
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tda_eeg import baseline, estilo, extracao, takens, topologia
from tda_eeg import config as cfg
from tda_eeg import dados

DESCR = {
    "h1_pers_max": "persistência máxima",
    "h1_pers_total": "persistência total",
    "h1_entropia": "entropia de persistência",
    "h1_n": "nº de classes em H1",
    "h1_pais2_l1": "norma L1 da paisagem λ2",
    "bottleneck": "bottleneck d_B(D_t, D_t-1)",
}
PRINCIPAIS = ["h1_pers_max", "h1_pers_total", "h1_entropia", "bottleneck"]
COLUNAS = topologia.COLUNAS_TOPO + ["bottleneck"] + baseline.COLUNAS_BASE


def auc(pos, neg):
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])


def blocos_interictais(df, col, k):
    """Média do descritor em todos os blocos de k janelas interictais consecutivas."""
    medias = []
    for _, sub in df.groupby("gravacao", sort=False):
        v = sub[col].to_numpy(float)
        ok = (sub.classe == "interictal").to_numpy() & np.isfinite(v)
        soma = np.convolve(np.where(ok, v, 0.0), np.ones(k), "valid")
        cont = np.convolve(ok.astype(float), np.ones(k), "valid")
        medias.append(soma[cont == k] / k)
    return np.concatenate(medias)


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------

def tabela_classes(df):
    com_crise = df.gravacao.isin(df.loc[df.classe == "ictal", "gravacao"].unique())
    linhas = []
    for c in COLUNAS:
        v = df[c].to_numpy(float)
        cl = df.classe.to_numpy()
        inter, inter_m = v[cl == "interictal"], v[(cl == "interictal") & com_crise.to_numpy()]
        linhas.append({
            "descritor": c,
            "grupo": "topologico" if c.startswith("h1_") or c == "bottleneck" else "classico",
            "mediana_interictal": np.nanmedian(inter),
            "mediana_pre_ictal": np.nanmedian(v[cl == "pre-ictal"]),
            "mediana_ictal": np.nanmedian(v[cl == "ictal"]),
            "auc_ictal_vs_inter": auc(v[cl == "ictal"], inter),
            "auc_ictal_vs_inter_mesma_grav": auc(v[cl == "ictal"], inter_m),
            "auc_pre_vs_inter": auc(v[cl == "pre-ictal"], inter),
            "auc_pre_vs_inter_mesma_grav": auc(v[cl == "pre-ictal"], inter_m),
            # observação pós-hoc (não prevista no plano): os 10 min após a crise
            "mediana_pos_ictal": np.nanmedian(v[cl == "pos-ictal"]),
            "p95_interictal": np.nanpercentile(inter, 95),
            "p95_pos_ictal": np.nanpercentile(v[cl == "pos-ictal"], 95),
            "auc_pos_vs_inter": auc(v[cl == "pos-ictal"], inter),
            "auc_pos_vs_inter_mesma_grav": auc(v[cl == "pos-ictal"], inter_m),
        })
    return pd.DataFrame(linhas)


def teste_blocos(df):
    """Quão incomum é o trecho ictal entre trechos interictais de mesma duração?

    Para cada crise, T = média do descritor nas k janelas ictais. A referência são
    todos os blocos de k janelas interictais consecutivas. p é bilateral e empírico.
    """
    linhas = []
    for grav, sub in df[df.classe == "ictal"].groupby("gravacao"):
        k = len(sub)
        for c in COLUNAS:
            nulo = blocos_interictais(df, c, k)
            t = float(np.nanmean(sub[c]))
            m = np.median(nulo)
            p = (1 + np.sum(np.abs(nulo - m) >= abs(t - m))) / (1 + len(nulo))
            linhas.append({"crise": grav, "n_janelas_ictais": k, "descritor": c,
                           "media_ictal": t, "mediana_blocos_interictais": m,
                           "percentil": float(np.mean(nulo <= t) * 100),
                           "p_empirico": float(p), "n_blocos": len(nulo)})
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def fig_detalhe(g, zoom, df, ref):
    """EEG e descritores em torno de uma crise (tempo relativo ao início anotado)."""
    ini, fim = g.crises[0]
    x, fs = dados.carregar_canal(g.caminho)
    z = zoom[zoom.gravacao == g.nome]
    a, b = z.t_ini.min(), z.t_fim.max()
    tz = (z.t_ini + cfg.JANELA_S / 2 - ini).to_numpy()
    p = df[(df.gravacao == g.nome) & (df.t_ini >= a) & (df.t_fim <= b)]
    tp = (p.t_ini + cfg.JANELA_S / 2 - ini).to_numpy()

    fig, eixos = plt.subplots(5, 1, figsize=(11, 10.5), sharex=True, constrained_layout=True,
                              gridspec_kw={"height_ratios": [1.25, 1, 1, 1, 1]})
    i0, i1 = int(a * fs), int(b * fs)
    eixos[0].plot(np.arange(i0, i1) / fs - ini, x[i0:i1], color=estilo.TINTA_2, lw=0.45)
    eixos[0].set_ylabel("EEG (µV)")
    eixos[0].set_title(f"{cfg.CANAL} filtrado em 1-30 Hz")
    for ax, c in zip(eixos[1:4], PRINCIPAIS[:3]):
        ax.axhspan(*ref[c], color=estilo.GRADE, alpha=0.7, lw=0, zorder=0)
        ax.plot(tz, z[c], color=estilo.AZUL, lw=1.5)
        ax.set_ylabel(DESCR[c])
        ax.set_title(f"H1: {DESCR[c]} (janelas de 8 s, passo de 1 s)")
    ax = eixos[4]
    ax.axhspan(*ref["bottleneck"], color=estilo.GRADE, alpha=0.7, lw=0, zorder=0)
    ax.plot(tp, p["bottleneck"], color=estilo.AZUL, lw=1.5, marker="o", ms=4)
    ax.set_ylabel("d_B")
    ax.set_title("Bottleneck entre janelas consecutivas (passo de 4 s)")
    ax.set_xlabel("tempo do centro da janela relativo ao início anotado da crise (s)")
    for k, ax in enumerate(eixos):
        estilo.faixa_crise(ax, 0, fim - ini, "crise anotada" if k == 0 else None)
    fig.suptitle(f"{g.nome}: crise em {ini:.0f}-{fim:.0f} s — descritores de H1 ao longo do tempo\n"
                 "faixa cinza = intervalo interquartil das janelas interictais (todas as gravações)",
                 fontsize=12)
    estilo.salvar(fig, cfg.DIR_FIG / f"04_detalhe_{g.nome}.png")


def fig_linha_do_tempo(gravs, df):
    cols = ["h1_pers_max", "h1_pers_total"]
    fig, eixos = plt.subplots(len(gravs), 2, figsize=(12, 1.55 * len(gravs) + 1), sharex=True,
                              sharey="col", constrained_layout=True)
    for i, g in enumerate(gravs):
        sub = df[df.gravacao == g.nome]
        t = (sub.t_ini + cfg.JANELA_S / 2) / 60
        for j, c in enumerate(cols):
            ax = eixos[i, j]
            ax.plot(t, sub[c], color=estilo.AZUL, lw=0.5, alpha=0.35)
            ax.plot(t, sub[c].rolling(15, center=True, min_periods=5).median(),
                    color=estilo.AZUL, lw=1.8)
            for ini, fim in g.crises:
                estilo.faixa_crise(ax, ini / 60, fim / 60)
                ax.axvline(ini / 60, color=estilo.LARANJA, lw=1.2)
                if j == 0:
                    ax.text(ini / 60, 0.96, " crise", transform=ax.get_xaxis_transform(),
                            ha="left", va="top", fontsize=8.5, color=estilo.TINTA_2)
            if j == 0:
                ax.set_ylabel(g.nome, rotation=0, ha="right", va="center", color=estilo.TINTA)
            if i == 0:
                ax.set_title(f"H1: {DESCR[c]} (fino: janelas; grosso: mediana móvel de 1 min)")
    for ax in eixos[-1]:
        ax.set_xlabel("tempo na gravação (min)")
    fig.suptitle(f"Sete horas de EEG ({cfg.CANAL}): os descritores de H1 em cada gravação")
    estilo.salvar(fig, cfg.DIR_FIG / "04_linha_do_tempo.png")


def fig_exemplos(g, df, diagramas):
    """Janela típica (persistência máxima mais próxima da mediana da classe) de cada classe."""
    x, fs = dados.carregar_canal(g.caminho)
    w = int(cfg.JANELA_S * fs)
    fig, eixos = plt.subplots(3, 3, figsize=(11.5, 9), constrained_layout=True,
                              gridspec_kw={"height_ratios": [0.7, 1.3, 1.3]})
    escolhidas, lim_d = [], 0.0
    for classe in cfg.CLASSES:
        sub = df[(df.gravacao == g.nome) & (df.classe == classe)]
        i = (sub.h1_pers_max - sub.h1_pers_max.median()).abs().idxmin()
        escolhidas.append((classe, i))
        lim_d = max(lim_d, diagramas[i][:, 1].max())
    lim_d *= 1.06
    for j, (classe, i) in enumerate(escolhidas):
        cor = estilo.COR_CLASSE[classe]
        t0 = df.loc[i, "t_ini"]
        seg = x[int(t0 * fs): int(t0 * fs) + w]
        X = takens.imersao_takens(seg, cfg.DIM, cfg.ATRASO, cfg.PASSO_NUVEM)
        h1 = diagramas[i]
        a, b, c = eixos[:, j]
        a.plot(np.arange(w) / fs, takens.padronizar(seg), color=cor, lw=1)
        a.set_title(f"{estilo.ROTULO_CLASSE[classe]} — {g.nome}, {t0:.0f}-{t0 + cfg.JANELA_S:.0f} s")
        a.set_xlabel("tempo na janela (s)")
        b.plot(X[:, 0], X[:, 1], color=estilo.GRADE, lw=0.7, zorder=1)
        b.scatter(X[:, 0], X[:, 1], s=13, color=cor, linewidths=0, zorder=2)
        b.set_aspect("equal", adjustable="datalim")
        b.set_title(f"Nuvem de Takens ({len(X)} pontos, d={cfg.DIM}, τ={cfg.ATRASO})")
        b.set_xlabel("x(t)")
        c.plot([0, lim_d], [0, lim_d], color=estilo.EIXO, lw=0.9)
        c.scatter(h1[:, 0], h1[:, 1], s=20, color=cor, linewidths=0, zorder=3)
        c.set_xlim(0, lim_d)
        c.set_ylim(0, lim_d)
        c.set_aspect("equal")
        c.set_title(f"H1: {len(h1)} classes, pers. máx. {df.loc[i, 'h1_pers_max']:.2f}, "
                    f"total {df.loc[i, 'h1_pers_total']:.2f}")
        c.set_xlabel("nascimento")
    eixos[0, 0].set_ylabel("x (padronizado)")
    eixos[1, 0].set_ylabel("x(t + τ)")
    eixos[2, 0].set_ylabel("morte")
    fig.suptitle("Janelas típicas de cada classe: sinal, nuvem de Takens e diagrama de H1")
    estilo.salvar(fig, cfg.DIR_FIG / "04_exemplos_por_classe.png")


def fig_distribuicoes(df, tab):
    rng = np.random.default_rng(cfg.SEMENTE)
    fig, eixos = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, (c, nome) in zip(eixos.ravel(), DESCR.items()):
        dados_c = [df.loc[df.classe == k, c].dropna().to_numpy() for k in cfg.CLASSES]
        bp = ax.boxplot(dados_c, widths=0.5, showfliers=False, patch_artist=True,
                        medianprops={"color": estilo.TINTA, "lw": 1.4},
                        whiskerprops={"color": estilo.MUDO}, capprops={"color": estilo.MUDO})
        for caixa, k in zip(bp["boxes"], cfg.CLASSES):
            caixa.set(facecolor=estilo.COR_CLASSE[k], alpha=0.45, edgecolor=estilo.COR_CLASSE[k])
        ict = dados_c[2]
        ax.scatter(3 + rng.uniform(-0.16, 0.16, len(ict)), ict, s=18, zorder=3,
                   color=estilo.COR_CLASSE["ictal"], edgecolor=estilo.SUPERFICIE, linewidths=0.6)
        ax.set_xticks([1, 2, 3], [f"{estilo.ROTULO_CLASSE[k]}\n(n={len(d)})"
                                  for k, d in zip(cfg.CLASSES, dados_c)])
        ax.grid(axis="x", visible=False)
        r = tab.set_index("descritor").loc[c]
        ax.set_title(nome)
        ax.set_xlabel(f"AUROC ictal×inter. {r.auc_ictal_vs_inter:.2f} | "
                      f"pré×inter. {r.auc_pre_vs_inter:.2f}", fontsize=8.5)
    fig.suptitle("Distribuição dos descritores de H1 por classe de janela (pontos: janelas ictais)")
    estilo.salvar(fig, cfg.DIR_FIG / "04_distribuicoes_por_classe.png")


def fig_paisagens(df, diagramas):
    t = np.linspace(*cfg.PAISAGEM_GRADE[:2], int(cfg.PAISAGEM_GRADE[2]))
    lam = np.stack([topologia.paisagens(d) for d in diagramas])       # (janela, k, grade)
    fig, eixos = plt.subplots(1, cfg.PAISAGEM_K, figsize=(12.5, 3.8), sharey=True,
                              constrained_layout=True)
    saida = {"t": t}
    for j, ax in enumerate(eixos):
        for classe in cfg.CLASSES:
            m = (df.classe == classe).to_numpy()
            media = lam[m, j].mean(0)
            ep = lam[m, j].std(0, ddof=1) / np.sqrt(m.sum())
            cor = estilo.COR_CLASSE[classe]
            ax.fill_between(t, media - 1.96 * ep, media + 1.96 * ep, color=cor, alpha=0.2, lw=0)
            ax.plot(t, media, color=cor, lw=2, label=f"{estilo.ROTULO_CLASSE[classe]} (n={m.sum()})")
            saida[f"lambda{j + 1}_{classe}"] = media
        # referência que expõe o efeito de gravação: interictal só das gravações com crise
        com_crise = df.gravacao.isin(df.loc[df.classe == "ictal", "gravacao"].unique())
        m = ((df.classe == "interictal") & com_crise).to_numpy()
        ax.plot(t, lam[m, j].mean(0), color=estilo.MUDO, lw=1.6, zorder=1,
                label=f"interictal, só gravações com crise (n={m.sum()})")
        ax.set_title(f"Paisagem média λ{j + 1}")
        ax.set_xlabel("escala da filtração")
        ax.set_xlim(0.2, 2.0)
    eixos[0].set_ylabel("λ(t)")
    eixos[-1].legend(loc="upper right")
    fig.suptitle("Paisagens de persistência médias por classe (faixa: ±1,96 erro-padrão, "
                 "janelas tratadas como independentes)")
    estilo.salvar(fig, cfg.DIR_FIG / "04_paisagens_medias.png")
    pd.DataFrame(saida).round(5).to_csv(cfg.DIR_TAB / "paisagens_medias.csv", index=False)


def transicoes(gravs, df):
    """Percentil (entre janelas interictais) do maior d_B perto do início e do fim da crise."""
    ref = df.loc[df.classe == "interictal", "bottleneck"].dropna().to_numpy()
    saida = []
    for g in gravs:
        sub = df[df.gravacao == g.nome]
        centro = sub.t_ini + cfg.JANELA_S / 2
        for ini, fim in g.crises:
            for nome, t in (("inicio", ini), ("fim", fim)):
                v = sub.loc[(centro - t).abs() <= cfg.JANELA_S, "bottleneck"].max()
                saida.append({"gravacao": g.nome, "evento": nome, "t": t, "dB_max": float(v),
                              "percentil_interictal": float(np.mean(ref <= v) * 100)})
    return pd.DataFrame(saida)


def main():
    cfg.garantir_pastas()
    estilo.aplicar()
    gravs = dados.listar_gravacoes()
    df, diagramas = extracao.carregar(extracao.rotulo_config(), com_diagramas=True)
    zoom = extracao.carregar("zoom")

    tab = tabela_classes(df)
    tab.round(4).to_csv(cfg.DIR_TAB / "estatisticas_classes.csv", index=False)
    blocos = teste_blocos(df)
    blocos.round(5).to_csv(cfg.DIR_TAB / "teste_blocos.csv", index=False)
    trans = transicoes(gravs, df)
    trans.round(4).to_csv(cfg.DIR_TAB / "bottleneck_transicoes.csv", index=False)

    pd.set_option("display.width", 220)
    print("\nAUROC por descritor (0,5 = sem separação):")
    print(tab[["descritor", "mediana_interictal", "mediana_pre_ictal", "mediana_ictal",
               "auc_ictal_vs_inter", "auc_ictal_vs_inter_mesma_grav", "auc_pre_vs_inter",
               "auc_pre_vs_inter_mesma_grav"]].round(3).to_string(index=False))
    print("\nTeste por blocos (trecho ictal x blocos interictais de mesma duração):")
    print(blocos[blocos.descritor.isin(PRINCIPAIS + ["h1_pais2_l1", "log_variancia"])]
          .round(4).to_string(index=False))
    print("\nBottleneck nas transições:")
    print(trans.round(3).to_string(index=False))

    resumo = {
        "n_janelas": df.classe.value_counts().to_dict(),
        "n_censuradas": int(df.h1_censuradas.sum()),
        "pontos_por_nuvem": int(df.n_pontos.iloc[0]),
    }
    (cfg.DIR_TAB / "resumo_amostra.json").write_text(json.dumps(resumo, indent=2), encoding="utf-8")

    inter = df[df.classe == "interictal"]
    ref = {c: tuple(inter[c].quantile([0.25, 0.75])) for c in PRINCIPAIS}
    for g in gravs:
        if g.crises:
            fig_detalhe(g, zoom, df, ref)
    fig_linha_do_tempo(gravs, df)
    fig_exemplos(next(g for g in gravs if g.crises), df, diagramas)
    fig_distribuicoes(df, tab)
    fig_paisagens(df, diagramas)


if __name__ == "__main__":
    main()
