"""Validação respeitando a dependência temporal: deixa-uma-gravação-fora.

Há um único paciente (chb01), então não é possível dividir treino/teste por
paciente como o enunciado recomenda para medir generalização entre indivíduos.
A unidade de dependência disponível é a GRAVAÇÃO (1 h contínua): todas as janelas
de uma gravação ficam juntas no treino ou no teste. Isso mede generalização para
outras horas e outra crise do MESMO paciente, não para pacientes novos.

Os hiperparâmetros dos modelos são fixos (não há busca), logo nada é ajustado
olhando o conjunto de teste.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as cfg


def modelos():
    return {
        "Regressão logística": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)),
        "Floresta aleatória": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, class_weight="balanced_subsample",
            random_state=cfg.SEMENTE, n_jobs=-1),
    }


def pontuar_fora_da_amostra(df, colunas, y, validas, modelo):
    """Escore de cada janela vindo de um modelo que NÃO viu a gravação dela.

    Treina só com as janelas `validas`; pontua todas as janelas da gravação retida.
    """
    X = df[colunas].to_numpy(float)
    grav = df.gravacao.to_numpy()
    escores = np.full(len(df), np.nan)
    for g in np.unique(grav):
        teste = grav == g
        treino = ~teste & validas
        if y[treino].sum() == 0:
            continue
        m = clone(modelo).fit(X[treino], y[treino])
        escores[teste] = m.predict_proba(X[teste])[:, 1]
    return escores


def sequencias(mask):
    """Número de sequências de valores True consecutivos."""
    mask = np.asarray(mask, dtype=bool)
    return int(mask[0]) + int(np.sum(mask[1:] & ~mask[:-1])) if len(mask) else 0
