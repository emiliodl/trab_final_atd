"""Tópico de ML: a topologia falha por modelo fraco, por falta de dados, ou por falta de informação?

Três testes, com os rótulos e os pacientes do script 12 (um canal, FP1-F7):
  1. modelos mais flexíveis (floresta aleatória, gradient boosting) no lugar da regressão logística;
  2. o cenário mais favorável possível: janelas sorteadas ao acaso entre treino e teste (com
     vazamento entre janelas vizinhas; serve só de teto);
  3. curva de aprendizado: treinar com 10 %, 25 %, 50 % e 100 % das janelas ictais de treino.

A AUROC só é calculada nos pacientes que têm as duas classes rotuladas (interictal a 1 h das crises).
Saída: resultados/multipaciente/tabelas/15_ml_*.csv.
"""
import _caminho  # noqa: F401
import importlib
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from tda_eeg import baseline, topologia
from tda_eeg import config as cfg
from tda_eeg import dados

m12 = importlib.import_module("12_avaliacao_supervisionada")
warnings.simplefilter("ignore")
DIR_TAB = cfg.DIR_MULTI / "tabelas"
CONJ = {"topologia (15)": topologia.COLUNAS_TOPO,
        "clássico só forma (9)": baseline.COLUNAS_FORMA,
        "clássico com amplitude (12)": baseline.COLUNAS_BASE}


def modelos():
    return {
        "regressão logística": m12.modelo(),
        "floresta aleatória": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=cfg.SEMENTE,
                                                     class_weight="balanced_subsample", n_jobs=-1),
        "gradient boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                                            random_state=cfg.SEMENTE),
    }


def main():
    gravs = dados.listar_gravacoes(pacientes=cfg.PACIENTES_MULTI, restringir=None)
    df = m12.rotular(m12.carregar(gravs), m12.crises_por_paciente(gravs))
    y = df.y_det.to_numpy()
    rot = np.isfinite(y)
    pac = df.paciente.to_numpy()
    # pacientes em que dá para medir a AUROC: têm janelas ictais e interictais rotuladas
    medir = [p for p in cfg.PACIENTES_MULTI if 0 < y[(pac == p) & rot].sum() < ((pac == p) & rot).sum()]
    print("AUROC medida em:", ", ".join(medir))

    def lopo(X, modelo, treino_extra=None):
        teste_auc, treino_auc = {}, []
        for p in medir:
            te = (pac == p) & rot
            tr = (pac != p) & rot if treino_extra is None else treino_extra(p)
            m = clone(modelo).fit(X[tr], y[tr])
            teste_auc[p] = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
            treino_auc.append(roc_auc_score(y[tr], m.predict_proba(X[tr])[:, 1]))
        return teste_auc, float(np.mean(treino_auc))

    # 1. modelos mais flexíveis
    linhas = []
    for nc, cols in CONJ.items():
        X = df[cols].to_numpy(float)
        for nm, mod in modelos().items():
            te, tr = lopo(X, mod)
            linhas.append({"descritores": nc, "modelo": nm, **{f"auroc_teste_{p}": v for p, v in te.items()},
                           "auroc_teste_media": float(np.mean(list(te.values()))), "auroc_treino_media": tr})
            print(f"  {nc:28s} {nm:20s} teste {linhas[-1]['auroc_teste_media']:.2f} | treino {tr:.2f}", flush=True)
    t1 = pd.DataFrame(linhas)
    t1.round(4).to_csv(DIR_TAB / "15_ml_modelos.csv", index=False)

    # 2. teto: janelas sorteadas (com vazamento)
    linhas = []
    for nc, cols in CONJ.items():
        X, yy = df[cols].to_numpy(float)[rot], y[rot]
        s = np.zeros(len(yy))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=cfg.SEMENTE).split(X, yy):
            s[te] = clone(modelos()["gradient boosting"]).fit(X[tr], yy[tr]).predict_proba(X[te])[:, 1]
        linhas.append({"descritores": nc, "auroc": roc_auc_score(yy, s), "auprc": average_precision_score(yy, s),
                       "prevalencia": float(yy.mean())})
    t2 = pd.DataFrame(linhas)
    t2.round(4).to_csv(DIR_TAB / "15_ml_teto_com_vazamento.csv", index=False)

    # 3. curva de aprendizado
    rng = np.random.default_rng(cfg.SEMENTE)
    linhas = []
    for nc, cols in CONJ.items():
        X = df[cols].to_numpy(float)
        for frac in (0.1, 0.25, 0.5, 1.0):
            medias = []
            for _ in range(5 if frac < 1 else 1):
                def sub(p, frac=frac):
                    tr = np.flatnonzero((pac != p) & rot)
                    pos = tr[y[tr] == 1]
                    fica = np.r_[tr[y[tr] == 0], rng.choice(pos, max(2, int(round(frac * len(pos)))), replace=False)]
                    m = np.zeros(len(df), bool)
                    m[fica] = True
                    return m
                te, _ = lopo(X, m12.modelo(), sub)
                medias.append(np.mean(list(te.values())))
            linhas.append({"descritores": nc, "fracao_ictais_treino": frac, "auroc_teste_media": float(np.mean(medias))})
    t3 = pd.DataFrame(linhas)
    t3.round(4).to_csv(DIR_TAB / "15_ml_curva_de_aprendizado.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n1. Modelos (teste no paciente que ficou de fora):\n" + t1.round(2).to_string(index=False))
    print("\n2. Teto com vazamento (janelas sorteadas, gradient boosting):\n" + t2.round(3).to_string(index=False))
    print("\n3. Curva de aprendizado (AUROC média no teste):\n" + t3.pivot(
        index="descritores", columns="fracao_ictais_treino", values="auroc_teste_media").round(3).to_string())


if __name__ == "__main__":
    main()
