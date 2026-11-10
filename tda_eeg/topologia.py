"""Fase 4: Vietoris-Rips, persistência em H1 e descritores dos diagramas."""
from __future__ import annotations

import gudhi
import numpy as np
from scipy.spatial.distance import pdist

from . import config as cfg

VAZIO = np.empty((0, 2))


def h1_rips(X, max_aresta=None, motor="gudhi"):
    """Persistência em H1 do Vietoris-Rips da nuvem X.

    Retorna (intervalos H1, corte, nº de classes censuradas). Sem `max_aresta`
    a filtração vai até a maior distância observada e nenhuma classe é
    censurada. Com truncamento, classes vivas no corte têm a morte fixada no
    corte e são contadas como censuradas (sua persistência é um limite inferior).

    motor="gudhi" segue o código-modelo do enunciado. motor="ripser" calcula a
    mesma persistência cerca de 25x mais rápido (diferença ~1e-7, conferida no
    script 08) e é usado apenas nas varreduras de sensibilidade.
    """
    X = np.asarray(X, dtype=float)
    corte_cheio = float(pdist(X).max())
    corte = corte_cheio if max_aresta is None else float(max_aresta)

    if motor == "ripser":
        from ripser import ripser
        h1 = np.asarray(ripser(X, maxdim=1, thresh=corte)["dgms"][1], dtype=float).reshape(-1, 2)
    else:
        rips = gudhi.RipsComplex(points=X, max_edge_length=corte)
        # dimensão 2: os triângulos são necessários para matar os ciclos de H1
        st = rips.create_simplex_tree(max_dimension=2)
        st.compute_persistence()
        h1 = np.asarray(st.persistence_intervals_in_dimension(1), dtype=float).reshape(-1, 2)

    censuradas = ~np.isfinite(h1[:, 1])
    h1[censuradas, 1] = corte
    return h1, corte, int(censuradas.sum())


def persistencias(h1):
    return (h1[:, 1] - h1[:, 0]) if len(h1) else np.empty(0)


def entropia_persistencia(h1):
    """E = -sum p_i log p_i, com p_i = persistência_i / persistência total."""
    p = persistencias(h1)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def paisagens(h1, k=cfg.PAISAGEM_K, grade=cfg.PAISAGEM_GRADE):
    """Paisagens de persistência lambda_1..lambda_k amostradas em uma grade fixa.

    lambda_j(t) é o j-ésimo maior valor de max(0, min(t - nasc, morte - t)).
    A grade é a mesma para todas as janelas (as nuvens vêm de sinais padronizados),
    o que torna as paisagens comparáveis e permite tirar médias por classe.
    """
    t = np.linspace(*grade[:2], int(grade[2]))
    saida = np.zeros((k, len(t)))
    if len(h1):
        tendas = np.clip(np.minimum(t[None, :] - h1[:, [0]], h1[:, [1]] - t[None, :]), 0, None)
        tendas = -np.sort(-tendas, axis=0)[:k]
        saida[: tendas.shape[0]] = tendas
    return saida


def descritores_h1(h1, corte=np.nan, n_censuradas=0):
    """Vetor de características de um diagrama de H1."""
    p = np.sort(persistencias(h1))[::-1]
    d = {
        "h1_n": len(p),
        "h1_pers_max": float(p[0]) if len(p) else 0.0,
        "h1_pers_total": float(p.sum()),
        "h1_pers_media": float(p.mean()) if len(p) else 0.0,
        "h1_entropia": entropia_persistencia(h1),
        # dominância: quanto da persistência total está no ciclo principal
        "h1_dominancia": float(p[0] / p.sum()) if len(p) and p.sum() > 0 else 0.0,
        "h1_pers_2a": float(p[1]) if len(p) > 1 else 0.0,
    }
    if len(h1):
        i = int(np.argmax(persistencias(h1)))
        d["h1_nasc_max"], d["h1_morte_max"] = float(h1[i, 0]), float(h1[i, 1])
    else:
        d["h1_nasc_max"] = d["h1_morte_max"] = 0.0

    lam = paisagens(h1)
    dt = (cfg.PAISAGEM_GRADE[1] - cfg.PAISAGEM_GRADE[0]) / (cfg.PAISAGEM_GRADE[2] - 1)
    for j in range(lam.shape[0]):
        d[f"h1_pais{j + 1}_l1"] = float(lam[j].sum() * dt)
        d[f"h1_pais{j + 1}_l2"] = float(np.sqrt((lam[j] ** 2).sum() * dt))
    d["corte"] = float(corte)
    d["h1_censuradas"] = int(n_censuradas)
    return d


COLUNAS_TOPO = [
    "h1_n", "h1_pers_max", "h1_pers_total", "h1_pers_media", "h1_entropia",
    "h1_dominancia", "h1_pers_2a", "h1_nasc_max", "h1_morte_max",
    "h1_pais1_l1", "h1_pais1_l2", "h1_pais2_l1", "h1_pais2_l2",
    "h1_pais3_l1", "h1_pais3_l2",
]


def bottleneck(h1_a, h1_b):
    """Distância de bottleneck entre dois diagramas de H1 (0 se ambos vazios)."""
    if len(h1_a) == 0 and len(h1_b) == 0:
        return 0.0
    return float(gudhi.bottleneck_distance(h1_a, h1_b))
