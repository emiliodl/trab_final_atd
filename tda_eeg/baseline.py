"""Baseline clássico: descritores de EEG por janela que não usam topologia.

Servem de comparação para os descritores topológicos (boas práticas, Seção 6.2).
Dois grupos:
  * dependentes da amplitude (variância, comprimento de linha, potência total);
  * dependentes apenas da forma do sinal (potências relativas, entropia espectral,
    Hjorth, cruzamentos por zero) -- comparação mais justa, pois a imersão de
    Takens padroniza cada janela e portanto descarta a amplitude.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import periodogram

BANDAS = {"delta": (1, 4), "teta": (4, 8), "alfa": (8, 13), "beta": (13, 30)}

COLUNAS_AMPLITUDE = ["log_variancia", "log_comp_linha", "log_potencia"]
COLUNAS_FORMA = [
    "rel_delta", "rel_teta", "rel_alfa", "rel_beta", "entropia_espectral",
    "freq_pico", "hjorth_mobilidade", "hjorth_complexidade", "cruz_zero",
]
COLUNAS_BASE = COLUNAS_AMPLITUDE + COLUNAS_FORMA


def descritores_classicos(seg, fs):
    seg = np.asarray(seg, dtype=float)
    x = seg - seg.mean()
    dx = np.diff(x)
    ddx = np.diff(dx)
    var, var_d, var_dd = x.var() + 1e-12, dx.var() + 1e-12, ddx.var() + 1e-12

    f, p = periodogram(x, fs=fs, window="hann")
    faixa = (f >= 1) & (f <= 30)
    f, p = f[faixa], p[faixa] + 1e-20
    total = p.sum()
    pn = p / total

    d = {
        "log_variancia": float(np.log(var)),
        "log_comp_linha": float(np.log(np.abs(dx).sum() + 1e-12)),
        "log_potencia": float(np.log(total)),
        "entropia_espectral": float(-(pn * np.log(pn)).sum() / np.log(len(pn))),
        "freq_pico": float(f[np.argmax(p)]),
        "hjorth_mobilidade": float(np.sqrt(var_d / var)),
        "hjorth_complexidade": float(np.sqrt(var_dd / var_d) / np.sqrt(var_d / var)),
        "cruz_zero": float(np.mean(np.signbit(x[:-1]) != np.signbit(x[1:])) * fs),
    }
    for nome, (lo, hi) in BANDAS.items():
        d[f"rel_{nome}"] = float(p[(f >= lo) & (f < hi)].sum() / total)
    return d
