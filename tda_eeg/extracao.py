"""Aplica Takens + Rips + H1 a todas as janelas de todas as gravações."""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config as cfg
from . import dados
from .baseline import descritores_classicos
from .takens import imersao_takens
from .topologia import bottleneck, descritores_h1, h1_rips


def rotulo_config(canal=cfg.CANAL, dim=cfg.DIM, atraso=cfg.ATRASO,
                  janela_s=cfg.JANELA_S, passo_s=cfg.PASSO_S):
    return f"{canal}_d{dim}_tau{atraso}_j{janela_s:g}_p{passo_s:g}"


def analisar_janela(seg, fs, dim=cfg.DIM, atraso=cfg.ATRASO, passo_nuvem=cfg.PASSO_NUVEM,
                    max_aresta=cfg.MAX_ARESTA, com_baseline=True, motor="gudhi"):
    """Descritores topológicos (e clássicos) de uma janela; devolve também o diagrama."""
    X = imersao_takens(seg, dim=dim, atraso=atraso, passo=passo_nuvem)
    h1, corte, cens = h1_rips(X, max_aresta, motor)
    d = descritores_h1(h1, corte, cens)
    d["n_pontos"] = len(X)
    if com_baseline:
        d.update(descritores_classicos(seg, fs))
    return d, h1


def extrair(gravacoes, canal=cfg.CANAL, dim=cfg.DIM, atraso=cfg.ATRASO,
            passo_nuvem=cfg.PASSO_NUVEM, janela_s=cfg.JANELA_S, passo_s=cfg.PASSO_S,
            max_aresta=cfg.MAX_ARESTA, com_baseline=True, com_bottleneck=True,
            recorte=None, motor="gudhi", n_jobs=-1, verbose=True):
    """Tabela de janelas x descritores e lista de diagramas H1 (mesma ordem).

    `recorte` = {gravacao: (t_min, t_max)} restringe a análise a um trecho.
    """
    janelas = dados.tabela_janelas(gravacoes, janela_s, passo_s)
    if recorte is not None:
        manter = np.zeros(len(janelas), dtype=bool)
        for nome, (a, b) in recorte.items():
            manter |= ((janelas.gravacao == nome) & (janelas.t_ini >= a) & (janelas.t_fim <= b)).to_numpy()
        janelas = janelas[manter].reset_index(drop=True)

    partes, diagramas = [], []
    with Parallel(n_jobs=n_jobs, batch_size=16) as par:
        for g in gravacoes:
            sub = janelas[janelas.gravacao == g.nome]
            if sub.empty:
                continue
            x, fs = dados.carregar_canal(g.caminho, canal)
            w = int(round(janela_s * fs))
            segs = [x[int(round(t * fs)): int(round(t * fs)) + w] for t in sub.t_ini]
            res = par(delayed(analisar_janela)(s, fs, dim, atraso, passo_nuvem, max_aresta,
                                               com_baseline, motor) for s in segs)
            feats = pd.DataFrame([r[0] for r in res], index=sub.index)
            diags = [r[1] for r in res]
            if com_bottleneck:
                # d_B(D_t, D_{t-1}) entre janelas consecutivas da mesma gravação
                db = par(delayed(bottleneck)(diags[i - 1], diags[i]) for i in range(1, len(diags)))
                feats["bottleneck"] = [np.nan] + list(db)
            partes.append(pd.concat([sub, feats], axis=1))
            diagramas.extend(diags)
            if verbose:
                print(f"  {g.nome} [{canal}] {len(sub)} janelas")
    df = pd.concat(partes).reset_index(drop=True)
    df.insert(1, "canal", canal)
    return df, diagramas


# --- cache em disco ---------------------------------------------------------

def salvar(df, diagramas, rotulo):
    cfg.garantir_pastas()
    df.to_csv(cfg.DIR_CACHE / f"descritores_{rotulo}.csv", index=False)
    if diagramas is not None:
        tam = np.array([len(d) for d in diagramas])
        pontos = np.concatenate([d for d in diagramas if len(d)]) if tam.sum() else np.empty((0, 2))
        np.savez_compressed(cfg.DIR_CACHE / f"diagramas_{rotulo}.npz", pontos=pontos, tam=tam)


def carregar(rotulo, com_diagramas=False):
    df = pd.read_csv(cfg.DIR_CACHE / f"descritores_{rotulo}.csv")
    if not com_diagramas:
        return df
    z = np.load(cfg.DIR_CACHE / f"diagramas_{rotulo}.npz")
    fim = np.cumsum(z["tam"])
    diagramas = [z["pontos"][a:b] for a, b in zip(fim - z["tam"], fim)]
    return df, diagramas


def extrair_ou_carregar(gravacoes, rotulo=None, refazer=False, **kw):
    chaves = {k: kw[k] for k in ("canal", "dim", "atraso", "janela_s", "passo_s") if k in kw}
    rotulo = rotulo or rotulo_config(**chaves)
    arq = cfg.DIR_CACHE / f"descritores_{rotulo}.csv"
    if arq.exists() and not refazer:
        return carregar(rotulo)
    df, diags = extrair(gravacoes, **kw)
    salvar(df, diags, rotulo)
    return df
