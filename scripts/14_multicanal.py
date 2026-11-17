"""Vários canais ao mesmo tempo: a topologia melhora quando o modelo vê os 21 canais?

O estudo de caso e os scripts 10-12 usam um canal (FP1-F7), como o enunciado pede. Aqui o
mesmo pipeline (Takens d=3, tau=4 -> Rips -> H1, janelas de 8 s) é aplicado a cada um dos 21
canais bipolares distintos, nos três pacientes, e os descritores entram juntos no modelo.

Duas formas de juntar os canais:
  * concatenados: uma coluna por (canal, descritor). O modelo pode aprender "onde" olhar,
    mas a posição do foco muda de paciente para paciente;
  * agregados: para cada descritor, média, máximo, mínimo e desvio-padrão entre os canais.
    Não depende de qual canal é qual, então se transfere melhor entre pacientes.

Rótulos, partições, limiar e métricas são os do script 12 (definição supervisionada).
A persistência é calculada com o Ripser (mesmos valores do GUDHI, cerca de 20 vezes mais rápido).

Saída: resultados/multipaciente/{tabelas,figuras}/14_*. Cache em resultados/cache/multicanal_*.npz.
"""
import _caminho  # noqa: F401
import importlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

from tda_eeg import baseline, estilo, topologia
from tda_eeg import config as cfg
from tda_eeg import dados
from tda_eeg.extracao import analisar_janela

m12 = importlib.import_module("12_avaliacao_supervisionada")

DIR_FIG = cfg.DIR_MULTI / "figuras"
DIR_TAB = cfg.DIR_MULTI / "tabelas"
COLUNAS = topologia.COLUNAS_TOPO + baseline.COLUNAS_BASE
FPRS = (0.01, 0.001)


def _janela_todos_canais(segs, fs):
    """Descritores de uma janela em todos os canais: matriz [canal, descritor]."""
    saida = np.empty((len(segs), len(COLUNAS)), dtype=np.float32)
    for c, seg in enumerate(segs):
        d, _ = analisar_janela(seg, fs, com_baseline=True, motor="ripser")
        saida[c] = [d[k] for k in COLUNAS]
    return saida


def extrair_paciente(pac, canais, refazer=False):
    """(tabela de janelas, cubo [janela, canal, descritor]) de um paciente, nos `canais` dados."""
    arq = cfg.DIR_CACHE / f"multicanal_{pac}.npz"
    gravs = dados.listar_gravacoes(pacientes=(pac,), restringir=None)
    janelas = dados.tabela_janelas(gravs)
    if arq.exists() and not refazer:
        z = np.load(arq, allow_pickle=True)
        if list(z["canais"]) == canais and len(z["cubo"]) == len(janelas):
            return janelas, z["cubo"]
    partes = []
    with Parallel(n_jobs=-1, batch_size=8) as par:
        for g in gravs:
            sub = janelas[janelas.gravacao == g.nome]
            X, fs = dados.carregar_canais(g.caminho, canais)          # filtra e reamostra os 21 de uma vez
            w = int(round(cfg.JANELA_S * fs))
            ini = [int(round(t * fs)) for t in sub.t_ini]
            partes += par(delayed(_janela_todos_canais)(X[:, i: i + w], fs) for i in ini)
            print(f"  {g.nome}: {len(sub)} janelas x {len(canais)} canais", flush=True)
    cubo = np.stack(partes)
    cfg.garantir_pastas()
    np.savez_compressed(arq, cubo=cubo, canais=np.array(canais, dtype=object))
    return janelas, cubo


# ---------------------------------------------------------------------------
# Conjuntos de descritores
# ---------------------------------------------------------------------------

def conjuntos(cubo, canais):
    """{nome: matriz [janela, coluna]}."""
    i_topo = [COLUNAS.index(c) for c in topologia.COLUNAS_TOPO]
    i_forma = [COLUNAS.index(c) for c in baseline.COLUNAS_FORMA]
    i_base = [COLUNAS.index(c) for c in baseline.COLUNAS_BASE]
    c0 = canais.index(cfg.CANAL)

    def concat(idx):
        return cubo[:, :, idx].reshape(len(cubo), -1)

    def agregado(idx):
        v = cubo[:, :, idx]
        return np.concatenate([v.mean(1), v.max(1), v.min(1), v.std(1)], axis=1)

    return {
        "Topologia, 1 canal (FP1-F7)": cubo[:, c0, i_topo],
        "Topologia, 21 canais concatenados": concat(i_topo),
        "Topologia, 21 canais agregados": agregado(i_topo),
        "Clássico só forma, 21 canais agregados": agregado(i_forma),
        "Clássico com amplitude, 1 canal (FP1-F7)": cubo[:, c0, i_base],
        "Clássico com amplitude, 21 canais agregados": agregado(i_base),
        "Topologia + clássico, 21 canais agregados": agregado(i_topo + i_base),
    }


def avaliar(df, X, crises):
    """Deixa-um-paciente-fora: métricas por janela e por evento em cada paciente testado."""
    y = df.y_det.to_numpy()
    rot = np.isfinite(y)
    linhas, detalhe = [], []
    for pac in cfg.PACIENTES_MULTI:
        teste = (df.paciente == pac).to_numpy()
        tr, te = ~teste & rot, teste & rot
        m = clone(m12.modelo()).fit(X[tr], y[tr])
        s = np.full(len(df), np.nan)
        s[teste] = m.predict_proba(X[teste])[:, 1]
        neg_tr = m.predict_proba(X[tr & (y == 0)])[:, 1]
        duas = 0 < y[te].sum() < te.sum()            # o paciente testado tem as duas classes rotuladas?
        r = {"paciente_teste": pac, "auroc": roc_auc_score(y[te], s[te]) if duas else np.nan,
             "auprc": average_precision_score(y[te], s[te]) if duas else np.nan,
             "auroc_treino": roc_auc_score(y[tr], m.predict_proba(X[tr])[:, 1])}
        for fpr in FPRS:
            ev, det = m12.eventos(df, teste, s, float(np.quantile(neg_tr, 1 - fpr)), crises[pac])
            r[f"crises_detectadas_{fpr:g}"] = ev["crises_detectadas"]
            r[f"fa_h_{fpr:g}"] = ev["fa_h"]
            r["crises_teste"] = ev["crises_teste"]
            detalhe += [{"paciente_teste": pac, "fpr_treino": fpr, **d} for d in det]
        linhas.append(r)
    return pd.DataFrame(linhas), pd.DataFrame(detalhe)


# ---------------------------------------------------------------------------
# Figura: uma bolinha por crise
# ---------------------------------------------------------------------------

def figura(detalhes, resumo, fpr=0.01):
    cores = estilo.COR_PACIENTE
    nomes = list(detalhes)
    ref = detalhes[nomes[0]]
    ref = ref[np.isclose(ref.fpr_treino, fpr)]
    crises = ref.crise.tolist()
    pac = [c[:5] for c in crises]
    x, pos = [], 0.0
    for i, p in enumerate(pac):
        if i and p != pac[i - 1]:
            pos += 0.9
        x.append(pos)
        pos += 1.0
    x = np.array(x)
    fig, ax = plt.subplots(figsize=(17, 8.6), constrained_layout=True)
    for j, nome in enumerate(nomes):
        yj = len(nomes) - 1 - j
        d = detalhes[nome]
        ok = d[np.isclose(d.fpr_treino, fpr)].set_index("crise").loc[crises].detectada.to_numpy(bool)
        for xi, p, det in zip(x, pac, ok):
            ax.scatter(xi, yj, s=200, facecolor=cores[p] if det else "none", edgecolor=cores[p], linewidths=2, zorder=3)
            if not det:
                ax.plot([xi - 0.2, xi + 0.2], [yj - 0.13, yj + 0.13], color=cores[p], lw=1.8, zorder=4)
                ax.plot([xi - 0.2, xi + 0.2], [yj + 0.13, yj - 0.13], color=cores[p], lw=1.8, zorder=4)
        r = resumo[resumo.descritores == nome].set_index("paciente_teste")
        ax.text(x.max() + 1.0, yj + 0.13, f"{ok.sum()} de {len(ok)}", ha="left", va="center", fontsize=23,
                fontweight="bold", color=estilo.TINTA)
        fa = r[f"fa_h_{fpr:g}"]
        ax.text(x.max() + 1.0, yj - 0.27, "alarmes falsos/h — " + " · ".join(
            f"{p[3:]}: " + (f"{fa[p]:.0f}" if fa[p] >= 1 or fa[p] < 0.05 else f"{fa[p]:.1f}".replace(".", ","))
            for p in cfg.PACIENTES_MULTI if np.isfinite(fa[p])), ha="left", va="center", fontsize=12.5,
            color=estilo.TINTA_2)
        ax.text(x.min() - 0.9, yj, nome, ha="right", va="center", fontsize=15, color=estilo.TINTA)
    for p in dict.fromkeys(pac):
        m = np.array([q == p for q in pac])
        ax.text(x[m].mean(), len(nomes) - 0.4, f"{p}\n{m.sum()} crise" + ("s" if m.sum() > 1 else ""), ha="center",
                va="bottom", fontsize=11.5, fontweight="bold", color=estilo.TINTA)
    ax.set_xlim(x.min() - 14.5, x.max() + 10.5)
    ax.set_ylim(-0.8, len(nomes) + 0.55)
    ax.axis("off")
    ax.text(x.min() - 0.3, -0.75, "bolinha cheia = crise detectada   ·   bolinha com × = crise perdida   ·   "
            "alarmes falsos medidos nos pacientes com interictal a 1 h das crises (01 a 04)", fontsize=12.5,
            color=estilo.TINTA_2, va="center")
    fig.suptitle("Um canal × 21 canais: treino nos outros pacientes, teste no que ficou de fora "
                 "(cada bolinha é uma crise)", fontsize=18)
    estilo.salvar(fig, DIR_FIG / "14_multicanal_crises.png")


def main():
    estilo.aplicar()
    DIR_FIG.mkdir(parents=True, exist_ok=True)
    DIR_TAB.mkdir(parents=True, exist_ok=True)
    refazer = "--refazer" in sys.argv
    t0 = time.time()
    # os 21 canais bipolares distintos do chb01; todos os pacientes usados têm esses canais
    canais = dados.nomes_canais(dados.listar_gravacoes(pacientes=("chb01",))[0].caminho)
    partes, cubos = [], []
    for pac in cfg.PACIENTES_MULTI:
        print(f"[{pac}]", flush=True)
        janelas, cubo = extrair_paciente(pac, canais, refazer)
        janelas.insert(0, "paciente", pac)
        janelas["canal"] = "21 canais"
        partes.append(janelas)
        cubos.append(cubo)
        print(f"  {len(janelas)} janelas | {time.time() - t0:.0f} s", flush=True)
    df = pd.concat(partes, ignore_index=True)
    cubo = np.concatenate(cubos)

    gravs = dados.listar_gravacoes(pacientes=cfg.PACIENTES_MULTI, restringir=None)
    crises = m12.crises_por_paciente(gravs)
    ordem = {g.nome: i for i, g in enumerate(gravs)}
    df["ordem_grav"] = df.gravacao.map(ordem)
    # as janelas já estão em ordem cronológica por paciente; m12.rotular só acrescenta os rótulos
    df = m12.rotular(df, crises)

    # separação por canal, um descritor de cada vez (para saber onde a crise aparece)
    y = df.y_det.to_numpy()
    linhas = []
    for pac in cfg.PACIENTES_MULTI:
        m = (df.paciente == pac).to_numpy() & np.isfinite(y)
        if not 0 < y[m].sum() < m.sum():                 # sem interictal a 1 h das crises
            continue
        for c, canal in enumerate(canais):
            linhas.append({"paciente": pac, "canal": canal,
                           "auroc_pers_max": roc_auc_score(y[m], cubo[m, c, COLUNAS.index("h1_pers_max")]),
                           "auroc_log_variancia": roc_auc_score(y[m], cubo[m, c, COLUNAS.index("log_variancia")])})
    por_canal = pd.DataFrame(linhas)
    por_canal.round(4).to_csv(DIR_TAB / "14_multicanal_por_canal.csv", index=False)
    print("\nAUROC por canal (ictal x interictal afastado), mínimo e máximo entre os 21 canais:")
    print(por_canal.groupby("paciente")[["auroc_pers_max", "auroc_log_variancia"]].agg(["min", "max"]).round(2).to_string())

    resumos, detalhes = [], {}
    for nome, X in conjuntos(cubo, canais).items():
        r, d = avaliar(df, np.asarray(X, dtype=float), crises)
        r.insert(0, "descritores", nome)
        r.insert(1, "n_colunas", X.shape[1])
        resumos.append(r)
        detalhes[nome] = d
        print(f"  {nome:45s} " + "  ".join(f"{a:.2f}" for a in r.auroc) + "  | crises (1 %): "
              + f"{int(r['crises_detectadas_0.01'].sum())}/{int(r.crises_teste.sum())}", flush=True)
    resumo = pd.concat(resumos, ignore_index=True)
    resumo.round(4).to_csv(DIR_TAB / "14_multicanal.csv", index=False)
    pd.concat([d.assign(descritores=n) for n, d in detalhes.items()]).to_csv(DIR_TAB / "14_multicanal_por_crise.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n" + resumo.round(2).to_string(index=False))
    figura(detalhes, resumo)
    print(f"\nTempo total: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
