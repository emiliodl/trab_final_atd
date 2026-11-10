"""Fase 3: imersão de Takens e critérios para escolher o atraso e a dimensão."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def padronizar(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-12)


def imersao_takens(x, dim=3, atraso=4, passo=4, padroniza=True):
    """v(t) = (x(t), x(t+tau), ..., x(t+(d-1)tau)), subamostrada de `passo` em `passo`."""
    x = padronizar(x) if padroniza else np.asarray(x, dtype=float)
    n = len(x) - (dim - 1) * atraso
    if n <= 0:
        raise ValueError("Sinal curto demais para dim/atraso escolhidos.")
    emb = np.column_stack([x[i * atraso: i * atraso + n] for i in range(dim)])
    return emb[::passo]


# --- Escolha do atraso ------------------------------------------------------

def autocorrelacao(x, max_lag):
    x = padronizar(x)
    n = len(x)
    return np.array([1.0] + [float(np.dot(x[:-k], x[k:]) / (n - k)) for k in range(1, max_lag + 1)])


def atraso_autocorrelacao(acf):
    """Primeiro lag com ACF < 1/e e primeiro cruzamento por zero (None se não houver)."""
    abaixo = np.flatnonzero(acf < 1 / np.e)
    zero = np.flatnonzero(acf <= 0)
    return (int(abaixo[0]) if len(abaixo) else None, int(zero[0]) if len(zero) else None)


def informacao_mutua(x, max_lag, bins=16):
    """Informação mútua média I(x(t); x(t+k)) em bits, por histograma equiprovável."""
    x = np.asarray(x, dtype=float)
    # bins com mesma massa: a estimativa não depende da escala/forma da amplitude
    postos = np.argsort(np.argsort(x))
    q = (postos * bins // len(x)).astype(int)
    ami = []
    for k in range(max_lag + 1):
        a, b = (q, q) if k == 0 else (q[:-k], q[k:])
        pxy = np.bincount(a * bins + b, minlength=bins * bins).reshape(bins, bins).astype(float)
        pxy /= pxy.sum()
        px, py = pxy.sum(1, keepdims=True), pxy.sum(0, keepdims=True)
        nz = pxy > 0
        ami.append(float(np.sum(pxy[nz] * np.log2(pxy[nz] / (px @ py)[nz]))))
    return np.array(ami)


def primeiro_minimo(curva):
    """Índice do primeiro mínimo local (None se a curva for monótona)."""
    for k in range(1, len(curva) - 1):
        if curva[k] < curva[k - 1] and curva[k] <= curva[k + 1]:
            return k
    return None


# --- Escolha da dimensão ----------------------------------------------------

def falsos_vizinhos(x, atraso, dims=range(1, 9), rtol=15.0, atol=2.0):
    """Fração de falsos vizinhos mais próximos (Kennel, Brown & Abarbanel, 1992).

    Um vizinho em dimensão d é falso se, ao passar para d+1, a distância cresce
    mais que `rtol` vezes, ou se a nova distância excede `atol` desvios-padrão.
    """
    x = padronizar(x)
    fracoes = []
    for d in dims:
        n = len(x) - d * atraso
        if n <= 10:
            fracoes.append(np.nan)
            continue
        emb = np.column_stack([x[i * atraso: i * atraso + n] for i in range(d)])
        dist, idx = cKDTree(emb).query(emb, k=2)
        r, viz = dist[:, 1], idx[:, 1]
        extra = np.abs(x[d * atraso: d * atraso + n] - x[viz + d * atraso])
        ok = r > 0
        falso = (extra[ok] / r[ok] > rtol) | (np.sqrt(r[ok] ** 2 + extra[ok] ** 2) > atol)
        fracoes.append(float(falso.mean()) if ok.any() else np.nan)
    return np.array(fracoes)


def dimensao_fnn(fracoes, dims, limiar=0.05):
    """Menor dimensão cuja fração de falsos vizinhos fica abaixo de `limiar`."""
    for d, f in zip(dims, fracoes):
        if np.isfinite(f) and f < limiar:
            return int(d)
    return None
