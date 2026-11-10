"""Tópico avançado - Previsão (EXPLORATÓRIO): há sinal pré-ictal nos descritores de H1?

Definições formais (fixadas antes de ver os resultados):
  * horizonte pré-ictal (SOP): 600 s antes do início anotado da crise;
  * tempo mínimo de intervenção (SPH): 0 s;
  * exclusão pós-ictal: 600 s após o fim da crise (fora do treino e das métricas);
  * janelas ictais e de transição: excluídas;
  * escore: classificador pré-ictal x interictal, deixa-uma-gravação-fora,
    suavizado por média móvel CAUSAL de 60 s (15 janelas);
  * alarme: escore suavizado >= limiar; depois de um alarme, período refratário de
    600 s sem novos alarmes;
  * acerto: alarme disparado dentro do horizonte pré-ictal;
  * alarme falso: alarme disparado em janela interictal; taxa = falsos / hora interictal;
  * referência de acaso: preditor aleatório de Poisson com a mesma taxa de alarmes,
    que acerta uma crise com probabilidade 1 - exp(-taxa x SOP).

Com duas crises a sensibilidade só pode ser 0, 50 ou 100 %: nada aqui é confirmatório.
"""
import _caminho  # noqa: F401
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tda_eeg import avaliacao, baseline, estilo, extracao, topologia
from tda_eeg import config as cfg
from tda_eeg import dados

SOP = cfg.HORIZONTE_PRE_S
REFRATARIO = 600.0
SUAVIZACAO = 15
CONJUNTOS = {
    "Topológico H1": topologia.COLUNAS_TOPO,
    "Clássico: forma + amplitude": baseline.COLUNAS_BASE,
    "Topológico + clássico": topologia.COLUNAS_TOPO + baseline.COLUNAS_BASE,
}
CORES = {"Topológico H1": estilo.AZUL, "Clássico: forma + amplitude": estilo.AGUA,
         "Topológico + clássico": estilo.VIOLETA}


def suavizar(df, escores, validas):
    s = pd.Series(np.where(validas, escores, np.nan))
    return (s.groupby(df.gravacao.to_numpy(), sort=False)
             .transform(lambda v: v.rolling(SUAVIZACAO, min_periods=1).mean()).to_numpy())


def alarmes(df, suave, validas, limiar):
    """Índices das janelas em que um alarme dispara (com período refratário)."""
    t = (df.t_abs + cfg.JANELA_S).to_numpy()
    disparos, ultimo = [], -np.inf
    for i in np.flatnonzero(validas & (suave >= limiar)):
        if t[i] - ultimo >= REFRATARIO:
            disparos.append(i)
            ultimo = t[i]
    return np.array(disparos, dtype=int)


def curva(df, suave, validas):
    classe = df.classe.to_numpy()
    crise = df.crise_proxima.to_numpy()
    horas = (classe == "interictal").sum() * cfg.PASSO_S / 3600
    linhas = []
    for limiar in np.unique(np.quantile(suave[validas], np.linspace(0, 1, 401))):
        idx = alarmes(df, suave, validas, limiar)
        pre = idx[classe[idx] == "pre-ictal"]
        linhas.append({"limiar": limiar, "crises_previstas": len(set(crise[pre])),
                       "alarmes_falsos": int((classe[idx] == "interictal").sum()),
                       "fa_h": (classe[idx] == "interictal").sum() / horas})
    return pd.DataFrame(linhas), horas


def main():
    cfg.garantir_pastas()
    estilo.aplicar()
    gravs = dados.listar_gravacoes()
    df = extracao.carregar(extracao.rotulo_config())
    validas = df.classe.isin(["interictal", "pre-ictal"]).to_numpy()
    y = (df.classe == "pre-ictal").to_numpy().astype(int)
    com_crise = [g.nome for g in gravs if g.crises]
    n_crises = len(com_crise)
    modelo = avaliacao.modelos()["Regressão logística"]

    resumo, suaves, curvas = [], {}, {}
    for nome, cols in CONJUNTOS.items():
        s = avaliacao.pontuar_fora_da_amostra(df, cols, y, validas, modelo)
        suave = suavizar(df, s, validas)
        suaves[nome] = suave
        c, horas = curva(df, suave, validas)
        curvas[nome] = c
        r = {"descritores": nome, "auroc_janelas": roc_auc_score(y[validas], s[validas])}
        for g in com_crise:
            m = validas & (df.gravacao == g).to_numpy()
            r[f"auroc_dentro_{g}"] = roc_auc_score(y[m], s[m])
        for k in range(1, n_crises + 1):
            ok = c[c.crises_previstas >= k]
            fa = float(ok.fa_h.min()) if len(ok) else np.nan
            r[f"fa_h_min_{k}_crise(s)"] = fa
            # preditor aleatório com a mesma taxa: acerta cada crise com probabilidade p
            p = 1 - np.exp(-fa * SOP / 3600) if np.isfinite(fa) else np.nan
            r[f"acaso_prob_{k}_crise(s)"] = 1 - (1 - p) ** n_crises if k == 1 else p ** k
        resumo.append(r)
    tab = pd.DataFrame(resumo)
    tab.round(4).to_csv(cfg.DIR_TAB / "previsao_resumo.csv", index=False)
    pd.concat([c.assign(descritores=n) for n, c in curvas.items()]).round(5).to_csv(
        cfg.DIR_TAB / "previsao_curvas.csv", index=False)
    (cfg.DIR_TAB / "previsao_definicoes.json").write_text(json.dumps({
        "horizonte_pre_ictal_s": SOP, "sph_s": 0, "exclusao_pos_ictal_s": cfg.EXCLUSAO_POS_S,
        "refratario_s": REFRATARIO, "suavizacao_janelas": SUAVIZACAO,
        "horas_interictais": horas, "n_crises": n_crises,
        "janelas_pre_ictais": int(y[validas].sum()), "janelas_interictais": int((y[validas] == 0).sum()),
        "validacao": "deixa-uma-gravacao-fora, regressao logistica com hiperparametros fixos",
    }, indent=2), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(f"Horas interictais: {horas:.2f} | crises: {n_crises}")
    print(tab.round(3).T.to_string())

    # --- figura 1: escore suavizado em todas as gravações ----------------------
    nomes = ["Topológico H1", "Clássico: forma + amplitude"]
    fig, eixos = plt.subplots(len(gravs), 2, figsize=(12, 1.35 * len(gravs) + 1.1), sharex=True,
                              sharey=True, constrained_layout=True)
    for i, g in enumerate(gravs):
        m = (df.gravacao == g.nome).to_numpy()
        t = ((df.t_ini[m] + cfg.JANELA_S) / 60).to_numpy()
        for j, nome in enumerate(nomes):
            ax = eixos[i, j]
            for ini, fim in g.crises:
                ax.axvspan((ini - SOP) / 60, ini / 60, color="#d6f1e6", lw=0, zorder=0)
                ax.axvline(ini / 60, color=estilo.LARANJA, lw=1.4)
                if j == 0:
                    ax.text((ini - SOP / 2) / 60, 1.0, "pré-ictal", ha="center", va="bottom",
                            transform=ax.get_xaxis_transform(), fontsize=8.5, color=estilo.TINTA_2)
            ax.plot(t, suaves[nome][m], color=CORES[nome], lw=1.5)
            ax.set_ylim(-0.03, 1.03)
            if j == 0:
                ax.set_ylabel(g.nome, rotation=0, ha="right", va="center", color=estilo.TINTA)
            if i == 0:
                ax.set_title(f"{nome}\n", fontsize=10.5)
    for ax in eixos[-1]:
        ax.set_xlabel("tempo na gravação (min)")
    fig.suptitle("Escore pré-ictal suavizado (60 s), fora da amostra — linha laranja: início da crise")
    estilo.salvar(fig, cfg.DIR_FIG / "06_previsao_escores.png")

    # --- figura 2: crises previstas x alarmes falsos por hora -------------------
    fig, ax = plt.subplots(figsize=(7.4, 4.3), constrained_layout=True)
    fa = np.linspace(0, 6, 200)
    ax.plot(fa, n_crises * (1 - np.exp(-fa * SOP / 3600)), color=estilo.MUDO, lw=1.4,
            label="acaso (nº esperado de crises)")
    for nome, c in curvas.items():
        c = c.sort_values(["fa_h", "crises_previstas"])
        melhor = c.groupby("fa_h", as_index=False).crises_previstas.max()
        melhor["crises_previstas"] = melhor.crises_previstas.cummax()
        # o clássico e o combinado coincidem: o primeiro vai mais grosso, por baixo
        ax.step(np.r_[melhor.fa_h, 6], np.r_[melhor.crises_previstas, melhor.crises_previstas.iloc[-1]],
                where="post", color=CORES[nome], label=nome,
                lw=5 if nome.startswith("Clássico") else 2, alpha=0.6 if nome.startswith("Clássico") else 1)
    ax.set_xlim(0, 6)
    ax.set_yticks(range(n_crises + 1))
    ax.set_xlabel("alarmes falsos por hora interictal")
    ax.set_ylabel(f"crises previstas (de {n_crises})")
    ax.set_title("Previsão: crises antecipadas × taxa de alarmes falsos")
    ax.legend(loc="lower right")
    estilo.salvar(fig, cfg.DIR_FIG / "06_previsao_alarmes.png")


if __name__ == "__main__":
    main()
