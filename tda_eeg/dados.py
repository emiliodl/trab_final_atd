"""Fase 1 e 2: carga dos EDF do CHB-MIT, anotações de crise e pré-processamento."""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import requests

from . import config as cfg


@dataclass
class Gravacao:
    nome: str                      # ex.: "chb01_03"
    caminho: Path
    inicio_abs: float              # início em s, relativo à 1ª gravação usada
    duracao: float                 # duração em s
    crises: list = field(default_factory=list)   # [(ini, fim)] em s no arquivo

    @property
    def crises_abs(self):
        return [(a + self.inicio_abs, b + self.inicio_abs) for a, b in self.crises]


def ler_resumo(caminho=cfg.RESUMO):
    """Lê chbXX-summary.txt e devolve {arquivo.edf: [(ini_s, fim_s), ...]}."""
    crises, atual, ini = {}, None, None
    for linha in Path(caminho).read_text(encoding="utf-8").splitlines():
        m = re.match(r"File Name:\s*(\S+)", linha)
        if m:
            atual = m.group(1)
            crises[atual] = []
            continue
        m = re.match(r"Seizure\s*\d*\s*Start Time:\s*(\d+)", linha)
        if m:
            ini = float(m.group(1))
            continue
        m = re.match(r"Seizure\s*\d*\s*End Time:\s*(\d+)", linha)
        if m and atual is not None and ini is not None:
            crises[atual].append((ini, float(m.group(1))))
            ini = None
    return crises


def baixar_edf(nome_edf, destino=cfg.DIR_DADOS):
    """Baixa um EDF do PhysioNet caso ele não esteja na pasta local."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / nome_edf
    if not caminho.exists():
        with requests.get(f"{cfg.URL_BASE}/{nome_edf}", stream=True, timeout=180) as r:
            r.raise_for_status()
            with caminho.open("wb") as f:
                for bloco in r.iter_content(chunk_size=1024 * 1024):
                    if bloco:
                        f.write(bloco)
    return caminho


def _abrir(caminho):
    with warnings.catch_warnings():
        # o CHB-MIT repete o rótulo T8-P8; o MNE apenas renumera os duplicados
        warnings.simplefilter("ignore", RuntimeWarning)
        return mne.io.read_raw_edf(caminho, preload=False, verbose=False)


def listar_gravacoes(pasta=cfg.DIR_DADOS):
    """Gravações .edf presentes na pasta, em ordem cronológica, com as crises."""
    crises = ler_resumo()
    brutas = []
    for caminho in sorted(Path(pasta).glob("*.edf")):
        raw = _abrir(caminho)
        brutas.append((caminho, raw.info["meas_date"], raw.n_times / raw.info["sfreq"]))
    if not brutas:
        raise FileNotFoundError(f"Nenhum .edf encontrado em {pasta}")
    brutas.sort(key=lambda t: t[1])
    t0 = brutas[0][1]
    return [
        Gravacao(
            nome=caminho.stem,
            caminho=caminho,
            inicio_abs=(data - t0).total_seconds(),
            duracao=dur,
            crises=crises.get(caminho.name, []),
        )
        for caminho, data, dur in brutas
    ]


def carregar_canais(caminho, canais=(cfg.CANAL,), fs=cfg.FS, banda=cfg.BANDA):
    """Lê o EDF, filtra em passa-faixa e reamostra. Retorna (X[canal, t] em uV, fs).

    O filtro é aplicado à gravação inteira antes de qualquer recorte, de modo
    que os transientes de borda ficam fora das janelas analisadas.
    """
    raw = _abrir(caminho)
    canais = list(canais)
    faltando = [c for c in canais if c not in raw.ch_names]
    if faltando:
        raise ValueError(f"Canais {faltando} não encontrados; exemplos: {raw.ch_names[:5]}")
    raw = raw.pick(canais).load_data()
    raw.filter(banda[0], banda[1], verbose=False)
    raw.resample(fs, verbose=False)
    return raw.get_data() * 1e6, float(raw.info["sfreq"])


def carregar_canal(caminho, canal=cfg.CANAL, fs=cfg.FS, banda=cfg.BANDA):
    X, fs = carregar_canais(caminho, [canal], fs=fs, banda=banda)
    return X[0], fs


def nomes_canais(caminho):
    """Canais bipolares distintos.

    Descarta o T8-P8 repetido, canais vazios e o P7-T7, que é o T7-P7 com sinal
    trocado (a nuvem de Takens de -x é a de x refletida: mesma persistência).
    """
    nomes = []
    for c in _abrir(caminho).ch_names:
        if c.endswith("-1") or c in ("-", ".", "P7-T7"):
            continue
        nomes.append(c)
    return nomes


def tabela_janelas(gravacoes, janela_s=cfg.JANELA_S, passo_s=cfg.PASSO_S):
    """Uma linha por janela de todas as gravações, com tempos locais e absolutos."""
    linhas = []
    for g in gravacoes:
        inicios = np.arange(0.0, g.duracao - janela_s + 1e-9, passo_s)
        for t in inicios:
            linhas.append((g.nome, t, t + janela_s, t + g.inicio_abs))
    df = pd.DataFrame(linhas, columns=["gravacao", "t_ini", "t_fim", "t_abs"])
    return rotular(df, gravacoes, janela_s)


def rotular(df, gravacoes, janela_s=cfg.JANELA_S,
            horizonte_pre=cfg.HORIZONTE_PRE_S, exclusao_pos=cfg.EXCLUSAO_POS_S,
            fracao_ictal=cfg.FRACAO_ICTAL):
    """Atribui a classe de cada janela usando o tempo absoluto (entre arquivos).

    ictal       >= `fracao_ictal` da janela dentro de uma crise
    transicao   sobreposição parcial (< fracao_ictal) com uma crise  -> excluída
    pre-ictal   janela termina em (início - horizonte_pre, início]
    pos-ictal   janela começa em [fim, fim + exclusao_pos)            -> excluída
    interictal  todo o resto
    """
    df = df.copy()
    a = df["t_abs"].to_numpy()
    b = a + janela_s
    centro = a + janela_s / 2
    classe = np.full(len(df), "interictal", dtype=object)
    crises = [c for g in gravacoes for c in g.crises_abs]
    nomes = [f"{g.nome}@{int(ini)}s" for g in gravacoes for ini, _ in g.crises]

    # ordem de precedência: pós-ictal < pré-ictal < transição < ictal
    for ini, fim in crises:
        classe[(a >= fim) & (a < fim + exclusao_pos)] = "pos-ictal"
    for ini, fim in crises:
        classe[(b <= ini) & (b > ini - horizonte_pre)] = "pre-ictal"
    for ini, fim in crises:
        sobre = np.clip(np.minimum(b, fim) - np.maximum(a, ini), 0, None) / janela_s
        classe[(sobre > 0) & (sobre < fracao_ictal)] = "transicao"
    for ini, fim in crises:
        sobre = np.clip(np.minimum(b, fim) - np.maximum(a, ini), 0, None) / janela_s
        classe[sobre >= fracao_ictal] = "ictal"

    df["classe"] = classe
    if crises:
        inis = np.array([c[0] for c in crises])
        # tempo do centro da janela relativo ao início da crise mais próxima
        idx = np.argmin(np.abs(centro[:, None] - inis[None, :]), axis=1)
        df["t_rel_crise"] = centro - inis[idx]
        df["crise_proxima"] = np.array(nomes, dtype=object)[idx]
        # tempo até a PRÓXIMA crise (inf se não houver crise futura)
        futuro = np.where(inis[None, :] >= b[:, None], inis[None, :] - b[:, None], np.inf)
        df["t_ate_crise"] = futuro.min(axis=1)
    else:
        df["t_rel_crise"] = np.nan
        df["crise_proxima"] = ""
        df["t_ate_crise"] = np.inf
    return df
