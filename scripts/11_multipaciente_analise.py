"""Multipaciente (chb01, chb02, chb03): a conclusão do chb01 se repete em outros pacientes?

Lê os caches do script 10. Mesmas definições da análise original:
  * AUROC ictal x interictal, com referência em todas as gravações do paciente e
    só nas gravações com crise ("mesmas gravações", a comparação justa);
  * percentil do trecho ictal entre blocos interictais de mesma duração;
  * regressão logística com hiperparâmetros fixos; alarmes falsos por hora no
    limiar que detecta todas as crises do paciente.
Novidade possível só com vários pacientes: validação DEIXA-UM-PACIENTE-FORA, a
divisão pedida no enunciado para medir generalização a indivíduos novos.

Saída em resultados/multipaciente/{figuras,tabelas}; nada da análise original muda.
"""
import _caminho  # noqa: F401
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

from tda_eeg import avaliacao, baseline, estilo, extracao, topologia
from tda_eeg import config as cfg
from tda_eeg import dados

DIR_FIG = cfg.DIR_MULTI / "figuras"
DIR_TAB = cfg.DIR_MULTI / "tabelas"

DESCR = {
    "h1_pers_max": "persistência máxima",
    "h1_pers_total": "persistência total",
    "h1_entropia": "entropia de persistência",
    "h1_pais2_l1": "paisagem λ2 (L1)",
    "bottleneck": "bottleneck d_B",
    "log_variancia": "log-variância (baseline)",
}
CONJUNTOS = {
    "Topológico H1": topologia.COLUNAS_TOPO,
    "Clássico: só forma": baseline.COLUNAS_FORMA,
    "Clássico: forma + amplitude": baseline.COLUNAS_BASE,
    "Topológico + clássico": topologia.COLUNAS_TOPO + baseline.COLUNAS_BASE,
}
COR_PAC, MARCA_PAC = estilo.COR_PACIENTE, estilo.MARCA_PACIENTE


def desloc(k):
    """Deslocamento vertical do k-ésimo paciente dentro de uma linha da figura."""
    n = len(cfg.PACIENTES_MULTI)
    return (k - (n - 1) / 2) * min(0.18, 0.72 / n)
VALIDAS = ["ictal", "interictal", "pre-ictal"]


def auc(pos, neg):
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])


def carregar():
    partes = []
    for pac in cfg.PACIENTES_MULTI:
        d = extracao.carregar(f"{pac}_{extracao.rotulo_config()}")
        d.insert(0, "paciente", pac)
        partes.append(d)
    return pd.concat(partes, ignore_index=True)


# ---------------------------------------------------------------------------
# Separação por descritor
# ---------------------------------------------------------------------------

def tabela_auroc(df):
    linhas = []
    for pac, sub in df.groupby("paciente"):
        cl = sub.classe.to_numpy()
        com_crise = sub.gravacao.isin(sub.loc[sub.classe == "ictal", "gravacao"].unique()).to_numpy()
        for c in DESCR:
            v = sub[c].to_numpy(float)
            inter, inter_m = v[cl == "interictal"], v[(cl == "interictal") & com_crise]
            linhas.append({
                "paciente": pac, "descritor": c,
                "n_ictal": int((cl == "ictal").sum()), "n_interictal": len(inter),
                "auc_ictal_todas": auc(v[cl == "ictal"], inter),
                "auc_ictal_mesmas": auc(v[cl == "ictal"], inter_m),
                "auc_pre_todas": auc(v[cl == "pre-ictal"], inter),
                "auc_pre_mesmas": auc(v[cl == "pre-ictal"], inter_m),
            })
    return pd.DataFrame(linhas)


def blocos_interictais(sub, col, k):
    medias = []
    for _, g in sub.groupby("gravacao", sort=False):
        v = g[col].to_numpy(float)
        ok = (g.classe == "interictal").to_numpy() & np.isfinite(v)
        if len(v) < k:
            continue
        soma = np.convolve(np.where(ok, v, 0.0), np.ones(k), "valid")
        cont = np.convolve(ok.astype(float), np.ones(k), "valid")
        medias.append(soma[cont == k] / k)
    return np.concatenate(medias)


def tabela_crises(df):
    """Por crise: AUROC contra o interictal da própria gravação e teste por blocos."""
    linhas = []
    for crise, ict in df[df.classe == "ictal"].groupby("crise_proxima", sort=False):
        pac = ict.paciente.iloc[0]
        sub = df[df.paciente == pac]
        grav = ict.gravacao.iloc[0]
        inter_g = sub[(sub.gravacao == grav) & (sub.classe == "interictal")]
        k = len(ict)
        for c in DESCR:
            nulo = blocos_interictais(sub, c, k)
            t = float(np.nanmean(ict[c]))
            linhas.append({
                "paciente": pac, "crise": crise, "n_janelas_ictais": k, "descritor": c,
                "auc_vs_inter_mesma_grav": auc(ict[c].to_numpy(float), inter_g[c].to_numpy(float)),
                "percentil_blocos": float(np.mean(nulo <= t) * 100), "n_blocos": len(nulo),
            })
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# Classificação
# ---------------------------------------------------------------------------

def pontuar(df, colunas, y, validas, grupo, modelo):
    """Escore fora da amostra: cada valor de `grupo` sai do treino quando é pontuado."""
    X = df[colunas].to_numpy(float)
    escores = np.full(len(df), np.nan)
    for g in np.unique(grupo):
        teste = grupo == g
        treino = ~teste & validas
        if y[treino].sum() == 0:
            continue
        escores[teste] = clone(modelo).fit(X[treino], y[treino]).predict_proba(X[teste])[:, 1]
    return escores


def alarmes_por_hora(sub, s, y, validas):
    """Alarmes falsos/h no limiar que detecta TODAS as crises do paciente."""
    neg = validas & (y == 0)
    horas = neg.sum() * cfg.PASSO_S / 3600
    picos = [np.nanmax(s[(sub.crise_proxima == c).to_numpy() & (y == 1)])
             for c in sub.loc[y == 1, "crise_proxima"].unique()]
    lim = min(picos)
    if not np.isfinite(lim) or horas == 0:
        return np.nan
    falsos = sum(avaliacao.sequencias((s >= lim)[m] & neg[m])
                 for m in [(sub.gravacao == g).to_numpy() for g in sub.gravacao.unique()])
    return falsos / horas


def classificacao(df):
    modelo = avaliacao.modelos()["Regressão logística"]
    validas = df.classe.isin(VALIDAS).to_numpy()
    y = (df.classe == "ictal").to_numpy().astype(int)
    linhas = []
    for nome_c, cols in CONJUNTOS.items():
        # (a) dentro do paciente: a análise original repetida em cada paciente
        s_dentro = np.full(len(df), np.nan)
        for pac in cfg.PACIENTES_MULTI:
            m = (df.paciente == pac).to_numpy()
            sub = df[m].reset_index(drop=True)
            s_dentro[m] = pontuar(sub, cols, y[m], validas[m], sub.gravacao.to_numpy(), modelo)
        # (b) entre pacientes: treina em dois, testa no terceiro
        s_entre = pontuar(df, cols, y, validas, df.paciente.to_numpy(), modelo)
        for esquema, s in (("dentro do paciente", s_dentro), ("entre pacientes", s_entre)):
            for pac in cfg.PACIENTES_MULTI:
                m = (df.paciente == pac).to_numpy()
                v = m & validas & np.isfinite(s)
                sub = df[m].reset_index(drop=True)
                # sem as duas classes pontuadas não há AUROC: paciente com uma só gravação (dentro do
                # paciente não há o que deixar de fora) ou com uma só gravação com crise
                ok = v.any() and 0 < y[v].sum() < v.sum()
                linhas.append({
                    "esquema": esquema, "descritores": nome_c, "paciente": pac,
                    "auroc": roc_auc_score(y[v], s[v]) if ok else np.nan,
                    "auprc": average_precision_score(y[v], s[v]) if ok else np.nan,
                    "prevalencia": y[v].mean() if v.any() else np.nan,
                    "fa_h_todas_crises": alarmes_por_hora(sub, s[m], y[m], validas[m] & np.isfinite(s[m])) if ok else np.nan,
                })
            print(f"  {esquema:20s} {nome_c:28s} " + "  ".join(
                f"{r['paciente']} {r['auroc']:.2f}" for r in linhas[-len(cfg.PACIENTES_MULTI):]), flush=True)
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def fig_auroc(tab):
    nomes = list(DESCR)
    yy = np.arange(len(nomes))[::-1]
    fig, eixos = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True, constrained_layout=True)
    for ax, col, tit in zip(eixos, ["auc_ictal_todas", "auc_ictal_mesmas"],
                            ["Referência: todas as gravações do paciente",
                             "Referência: só as gravações com crise (comparação justa)"]):
        ax.axvline(0.5, color=estilo.EIXO, lw=1.2)
        for k, pac in enumerate(cfg.PACIENTES_MULTI):
            v = [tab[(tab.paciente == pac) & (tab.descritor == n)][col].item() for n in nomes]
            ax.scatter(v, yy - desloc(k), s=55, marker=MARCA_PAC[pac], color=COR_PAC[pac],
                       edgecolor=estilo.SUPERFICIE, linewidths=0.8, zorder=3, label=pac)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("AUROC ictal × interictal (0,5 = acaso)")
        ax.set_title(tit)
        ax.grid(axis="y", visible=False)
    eixos[0].set_yticks(yy, [DESCR[n] for n in nomes])
    eixos[1].legend(loc="lower left", title="paciente")
    fig.suptitle("Separação ictal × interictal por descritor, em cada paciente (FP1-F7, d=3, τ=4)")
    estilo.salvar(fig, DIR_FIG / "10_auroc_por_paciente.png")


def percentil_interictal(sub, col):
    ref = np.sort(sub.loc[sub.classe == "interictal", col].dropna().to_numpy(float))
    return 100 * np.searchsorted(ref, sub[col].to_numpy(float), side="right") / len(ref)


def fig_crises(df, gravs):
    crises = [(g, a, b) for g in gravs for a, b in g.crises]
    pacs = list(cfg.PACIENTES_MULTI)
    ncol = max(sum(g.paciente == p for g, _, _ in crises) for p in pacs)
    fig, eixos = plt.subplots(len(pacs), ncol, figsize=(3.4 * ncol, 2.3 * len(pacs)),
                              sharex=True, sharey=True, constrained_layout=True)
    perc = {}
    for pac in pacs:
        sub = df[df.paciente == pac]
        perc[pac] = {c: pd.Series(percentil_interictal(sub, c), index=sub.index)
                     for c in ("h1_pers_max", "log_variancia")}
    for i, pac in enumerate(pacs):
        cs = [(g, a, b) for g, a, b in crises if g.paciente == pac]
        for j in range(ncol):
            ax = eixos[i, j]
            if j >= len(cs):
                ax.set_visible(False)
                continue
            g, a, b = cs[j]
            ini = g.inicio_abs + a
            sub = df[df.paciente == pac]
            t = (sub.t_abs + cfg.JANELA_S / 2 - ini).to_numpy()
            perto = (t > -300) & (t < 300)
            ax.axhspan(25, 75, color=estilo.GRADE, alpha=0.7, lw=0, zorder=0)
            estilo.faixa_crise(ax, 0, b - a)
            ax.plot(t[perto], perc[pac]["log_variancia"].to_numpy()[perto], color=estilo.TINTA_2,
                    lw=1.1, label="log-variância")
            ax.plot(t[perto], perc[pac]["h1_pers_max"].to_numpy()[perto], color=estilo.AZUL,
                    lw=1.5, label="persistência máx. H1")
            ax.set_title(f"{g.nome} · crise de {b - a:.0f} s", fontsize=10)
            ax.set_ylim(-2, 102)
            if j == 0:
                ax.set_ylabel(f"{pac}\npercentil interictal")
            if i == len(pacs) - 1 or j >= sum(gg.paciente == pacs[i + 1] for gg, _, _ in crises):
                ax.set_xlabel("tempo relativo ao início (s)")
    eixos[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"As {len(crises)} crises: percentil de cada janela entre as janelas interictais do mesmo paciente "
                 "(faixa laranja = crise; cinza = percentis 25-75)")
    estilo.salvar(fig, DIR_FIG / "10_crises_percentis.png")


def fig_classificacao(clf):
    nomes = list(CONJUNTOS)
    yy = np.arange(len(nomes))[::-1]
    fig, eixos = plt.subplots(1, 2, figsize=(12.5, 4.0), sharey=True, constrained_layout=True)
    for ax, esq, tit in zip(eixos, ["dentro do paciente", "entre pacientes"],
                            ["Dentro do paciente (deixa-uma-gravação-fora)",
                             "Entre pacientes (deixa-um-paciente-fora)"]):
        ax.axvline(0.5, color=estilo.EIXO, lw=1.2)
        for k, pac in enumerate(cfg.PACIENTES_MULTI):
            v = np.array([clf[(clf.esquema == esq) & (clf.paciente == pac) & (clf.descritores == n)].auroc.item()
                          for n in nomes])
            if not np.isfinite(v).any():
                continue
            ax.scatter(v, yy - desloc(k), s=60, marker=MARCA_PAC[pac], color=COR_PAC[pac],
                       edgecolor=estilo.SUPERFICIE, linewidths=0.8, zorder=3, label=pac)
        ax.set_xlim(0.3, 1.02)
        ax.set_xlabel("AUROC no paciente testado (0,5 = acaso)")
        ax.set_title(tit)
        ax.grid(axis="y", visible=False)
    eixos[0].set_yticks(yy, nomes)
    eixos[1].legend(loc="lower left", title="paciente testado")
    fig.suptitle("Detecção de janelas ictais com regressão logística: topologia × baselines clássicos")
    estilo.salvar(fig, DIR_FIG / "10_classificacao_pacientes.png")


def fig_linha_do_tempo(df, gravs):
    pacs = list(cfg.PACIENTES_MULTI)
    fig, eixos = plt.subplots(len(pacs), 1, figsize=(13, 2.6 * len(pacs)), constrained_layout=True)
    for ax, pac in zip(eixos, pacs):
        for g in (g for g in gravs if g.paciente == pac):
            s = df[df.gravacao == g.nome].sort_values("t_ini")
            h = (s.t_abs + cfg.JANELA_S / 2).to_numpy() / 3600
            ax.plot(h, s.h1_pers_total, color=estilo.AZUL, lw=0.4, alpha=0.35)
            ax.plot(h, s.h1_pers_total.rolling(15, center=True, min_periods=5).median(),
                    color=estilo.AZUL, lw=1.6)
            for a, _ in g.crises:
                ax.axvline((g.inicio_abs + a) / 3600, color=estilo.LARANJA, lw=1.8)
        ax.set_title(f"{pac}: persistência total de H1 (fino: janelas; grosso: mediana móvel de 1 min; "
                     "laranja: crise)", fontsize=10)
        ax.set_ylabel("persistência total")
    eixos[-1].set_xlabel("horas desde o início da primeira gravação do paciente")
    estilo.salvar(fig, DIR_FIG / "10_linha_do_tempo.png")


def main():
    estilo.aplicar()
    DIR_FIG.mkdir(parents=True, exist_ok=True)
    DIR_TAB.mkdir(parents=True, exist_ok=True)
    gravs = dados.listar_gravacoes(pacientes=cfg.PACIENTES_MULTI, restringir=None)
    df = carregar()

    resumo = (df.groupby("paciente").agg(gravacoes=("gravacao", "nunique"), janelas=("t_ini", "size"))
              .join(df.pivot_table(index="paciente", columns="classe", values="t_ini", aggfunc="size")))
    resumo["horas"] = resumo.janelas * cfg.PASSO_S / 3600
    resumo["crises"] = [sum(len(g.crises) for g in gravs if g.paciente == p) for p in resumo.index]
    resumo.to_csv(DIR_TAB / "resumo_pacientes.csv")
    print(resumo.round(1).to_string(), "\n")

    tab = tabela_auroc(df)
    tab.round(4).to_csv(DIR_TAB / "auroc_por_paciente.csv", index=False)
    print(tab.round(2).to_string(index=False), "\n")

    cr = tabela_crises(df)
    cr.round(4).to_csv(DIR_TAB / "auroc_por_crise.csv", index=False)
    print(cr.pivot_table(index="crise", columns="descritor", values="percentil_blocos")
          .round(0).to_string(), "\n")

    clf = classificacao(df)
    clf.round(4).to_csv(DIR_TAB / "classificacao_pacientes.csv", index=False)
    print("\n" + clf.round(2).to_string(index=False))

    fig_auroc(tab)
    fig_crises(df, gravs)
    fig_classificacao(clf)
    fig_linha_do_tempo(df, gravs)

    (DIR_TAB / "configuracao.json").write_text(json.dumps({
        "pacientes": list(cfg.PACIENTES_MULTI), "canal": cfg.CANAL, "d": cfg.DIM, "tau": cfg.ATRASO,
        "janela_s": cfg.JANELA_S, "passo_s": cfg.PASSO_S, "horizonte_pre_s": cfg.HORIZONTE_PRE_S,
        "exclusao_pos_s": cfg.EXCLUSAO_POS_S, "fracao_ictal": cfg.FRACAO_ICTAL,
        "modelo": "regressao logistica, C=1, class_weight=balanced (hiperparametros fixos)",
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
