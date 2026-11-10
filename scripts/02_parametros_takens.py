"""Fase 3 (análise completa): escolha de tau e d por autocorrelação, informação
mútua média e falsos vizinhos.

A escolha usa SOMENTE trechos interictais da gravação chb01_01 (sem crise), ou
seja, não usa os rótulos nem o desempenho de nenhum classificador. As curvas dos
trechos ictais são mostradas apenas para descrição.
"""
import _caminho  # noqa: F401
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tda_eeg import config as cfg
from tda_eeg import dados, estilo, takens

MAX_LAG = 40          # 40 amostras = 625 ms a 64 Hz
DIMS = list(range(1, 9))
TRECHO_S = 60.0
N_TRECHOS = 30
LIMIAR_FNN = 0.10     # fixado a priori; EEG ruidoso não chega a 0 % de falsos vizinhos


def curvas(segmentos, atraso_fnn):
    acf = np.array([takens.autocorrelacao(s, MAX_LAG) for s in segmentos])
    ami = np.array([takens.informacao_mutua(s, MAX_LAG) for s in segmentos])
    fnn = np.array([takens.falsos_vizinhos(s, atraso_fnn, DIMS) for s in segmentos])
    return acf, ami, fnn


def main():
    cfg.garantir_pastas()
    estilo.aplicar()
    gravs = {g.nome: g for g in dados.listar_gravacoes()}

    x, fs = dados.carregar_canal(gravs["chb01_01"].caminho)
    n = int(TRECHO_S * fs)
    inicios = np.linspace(0, len(x) - n, N_TRECHOS).astype(int)
    inter = [x[i: i + n] for i in inicios]

    ictais = []
    for g in gravs.values():
        if g.crises:
            xg, _ = dados.carregar_canal(g.caminho)
            ictais += [xg[int(a * fs): int(b * fs)] for a, b in g.crises]

    # 1) atraso: ACF e informação mútua em cada trecho interictal
    acf_i = np.array([takens.autocorrelacao(s, MAX_LAG) for s in inter])
    ami_i = np.array([takens.informacao_mutua(s, MAX_LAG) for s in inter])
    tau_e = [takens.atraso_autocorrelacao(a)[0] for a in acf_i]
    tau_0 = [takens.atraso_autocorrelacao(a)[1] for a in acf_i]
    tau_ami = [takens.primeiro_minimo(a) for a in ami_i]

    def mediana(v):
        v = [u for u in v if u is not None]
        return int(np.median(v)) if v else None

    atraso = mediana(tau_ami) or mediana(tau_e)

    # 2) dimensão: falsos vizinhos usando o atraso escolhido
    acf_i, ami_i, fnn_i = curvas(inter, atraso)
    acf_c, ami_c, fnn_c = curvas(ictais, atraso)
    fnn_med = np.nanmedian(fnn_i, axis=0)
    dim = takens.dimensao_fnn(fnn_med, DIMS, LIMIAR_FNN) or DIMS[-1]

    resumo = {
        "canal": cfg.CANAL, "fs": fs, "n_trechos_interictais": len(inter),
        "trecho_s": TRECHO_S,
        "atraso_acf_1e_mediana": mediana(tau_e),
        "atraso_acf_zero_mediana": mediana(tau_0),
        "atraso_ami_mediana": mediana(tau_ami),
        "trechos_sem_minimo_ami": int(sum(t is None for t in tau_ami)),
        "atraso_sugerido": atraso,
        "atraso_sugerido_ms": 1000 * atraso / fs,
        "limiar_fnn": LIMIAR_FNN,
        "fnn_mediana_por_dim": {str(d): float(f) for d, f in zip(DIMS, fnn_med)},
        "dim_sugerida": dim,
        "enunciado": {"dim": cfg.DIM, "atraso": cfg.ATRASO},
    }
    (cfg.DIR_TAB / "parametros_takens.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame({
        "lag": np.arange(MAX_LAG + 1),
        "acf_interictal": np.median(acf_i, 0), "acf_ictal": np.median(acf_c, 0),
        "ami_interictal": np.median(ami_i, 0), "ami_ictal": np.median(ami_c, 0),
    }).to_csv(cfg.DIR_TAB / "parametros_takens_curvas.csv", index=False)
    print(json.dumps(resumo, indent=2, ensure_ascii=False))

    # --- figura -------------------------------------------------------------
    lags_ms = 1000 * np.arange(MAX_LAG + 1) / fs
    fig, eixos = plt.subplots(1, 3, figsize=(12.5, 3.7), constrained_layout=True)
    paineis = [
        (eixos[0], lags_ms, acf_i, acf_c, "Autocorrelação", "atraso (ms)"),
        (eixos[1], lags_ms, ami_i, ami_c, "Informação mútua média (bits)", "atraso (ms)"),
        (eixos[2], np.array(DIMS), fnn_i * 100, fnn_c * 100,
         f"Falsos vizinhos (%), τ = {atraso}", "dimensão de imersão d"),
    ]
    ci, cc = estilo.COR_CLASSE["interictal"], estilo.COR_CLASSE["ictal"]
    for ax, xs, inter_c, ict_c, titulo, xl in paineis:
        q1, q3 = np.nanpercentile(inter_c, [25, 75], axis=0)
        ax.fill_between(xs, q1, q3, color=ci, alpha=0.18, lw=0)
        ax.plot(xs, np.nanmedian(inter_c, 0), color=ci, lw=2, label="interictal (mediana, IQR)")
        ax.plot(xs, np.nanmedian(ict_c, 0), color=cc, lw=2, label="ictal (2 crises)")
        ax.set_title(titulo)
        ax.set_xlabel(xl)
    eixos[0].axhline(1 / np.e, color=estilo.MUDO, lw=0.8)
    eixos[0].text(lags_ms[-1], 1 / np.e, "1/e", ha="right", va="bottom", fontsize=8.5,
                  color=estilo.TINTA_2)
    eixos[2].axhline(100 * LIMIAR_FNN, color=estilo.MUDO, lw=0.8)
    eixos[2].text(DIMS[-1], 100 * LIMIAR_FNN, f"limiar {100 * LIMIAR_FNN:.0f} %", ha="right",
                  va="bottom", fontsize=8.5, color=estilo.TINTA_2)
    for ax, v, txt in [(eixos[0], 1000 * cfg.ATRASO / fs, "τ enunciado"),
                       (eixos[1], 1000 * atraso / fs, "τ sugerido")]:
        ax.axvline(v, color=estilo.TINTA_2, lw=0.9)
        ax.text(v, 0.97, f" {txt}", transform=ax.get_xaxis_transform(), ha="left", va="top",
                fontsize=8.5, color=estilo.TINTA_2)
    eixos[2].axvline(dim, color=estilo.TINTA_2, lw=0.9)
    eixos[2].text(dim, 0.97, " d sugerida", transform=eixos[2].get_xaxis_transform(),
                  ha="left", va="top", fontsize=8.5, color=estilo.TINTA_2)
    eixos[1].set_ylim(0, 1.5)      # o valor em atraso 0 (entropia do sinal) fica fora da escala
    eixos[0].legend(loc="upper right")
    fig.suptitle(f"Escolha dos parâmetros da imersão de Takens ({cfg.CANAL}, {fs:.0f} Hz)")
    estilo.salvar(fig, cfg.DIR_FIG / "02_parametros_takens.png")


if __name__ == "__main__":
    main()
