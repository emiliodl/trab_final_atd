"""Validação do pipeline em sinais sintéticos com resposta conhecida.

Um sinal periódico vira um laço no espaço de Takens (uma classe dominante em H1);
ruído sem estrutura não. Antes de usar o EEG, conferimos se o código reproduz isso
e como a persistência máxima cai quando o ruído aumenta.
"""
import _caminho  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from tda_eeg import config as cfg
from tda_eeg import estilo, takens, topologia

FS = cfg.FS
T = np.arange(int(cfg.JANELA_S * FS)) / FS


def ruido_banda(rng, n):
    """Ruído branco filtrado em 1-30 Hz, como o EEG pré-processado."""
    sos = butter(4, cfg.BANDA, btype="band", fs=FS, output="sos")
    return sosfiltfilt(sos, rng.standard_normal(n + 4 * FS))[2 * FS: 2 * FS + n]


def espicula_onda(t, f=3.0):
    """Padrão periódico não senoidal, lembrando descargas espícula-onda a 3 Hz."""
    fase = (t * f) % 1.0
    return np.exp(-((fase - 0.15) / 0.035) ** 2) * 2.2 - np.sin(2 * np.pi * fase) * 0.8


def sinais(rng):
    seno = np.sin(2 * np.pi * 3.0 * T)
    r = takens.padronizar(ruido_banda(rng, len(T)))
    return {
        "Senoide 3 Hz": seno,
        "Senoide 3 Hz + ruído (SNR 0 dB)": takens.padronizar(seno) + r,
        "Espícula-onda sintética 3 Hz": espicula_onda(T),
        "Ruído 1-30 Hz": r,
    }


def main():
    cfg.garantir_pastas()
    estilo.aplicar()
    rng = np.random.default_rng(cfg.SEMENTE)

    # --- figura 1: sinal -> nuvem -> diagrama --------------------------------
    S = sinais(rng)
    fig, eixos = plt.subplots(len(S), 3, figsize=(11.5, 2.55 * len(S)),
                              gridspec_kw={"width_ratios": [2.4, 1, 1]}, constrained_layout=True)
    linhas = []
    for i, (nome, x) in enumerate(S.items()):
        X = takens.imersao_takens(x, cfg.DIM, cfg.ATRASO, cfg.PASSO_NUVEM)
        h1, corte, _ = topologia.h1_rips(X)
        d = topologia.descritores_h1(h1, corte)
        linhas.append({"sinal": nome, **{k: d[k] for k in
                       ("h1_n", "h1_pers_max", "h1_pers_total", "h1_entropia", "h1_dominancia")}})
        a, b, c = eixos[i]
        a.plot(T, takens.padronizar(x), color=estilo.AZUL, lw=1.1)
        a.set_title(nome)
        a.set_ylabel("x (padronizado)")
        b.plot(X[:, 0], X[:, 1], color=estilo.GRADE, lw=0.7, zorder=1)
        b.scatter(X[:, 0], X[:, 1], s=9, color=estilo.AZUL, zorder=2, linewidths=0)
        b.set_aspect("equal", adjustable="datalim")
        b.set_title("Nuvem de Takens")
        b.set_ylabel("x(t + τ)")
        lim = max(1.0, h1[:, 1].max() * 1.08) if len(h1) else 1.0
        c.plot([0, lim], [0, lim], color=estilo.EIXO, lw=0.9)
        if len(h1):
            c.scatter(h1[:, 0], h1[:, 1], s=16, color=estilo.LARANJA, linewidths=0, zorder=3)
        c.set_xlim(0, lim)
        c.set_ylim(0, lim)
        c.set_aspect("equal")
        c.set_title(f"H1 — persistência máx. {d['h1_pers_max']:.2f}")
        c.set_ylabel("morte")
    eixos[-1, 0].set_xlabel("tempo (s)")
    eixos[-1, 1].set_xlabel("x(t)")
    eixos[-1, 2].set_xlabel("nascimento")
    fig.suptitle("Validação sintética: periodicidade aparece como um ciclo persistente em H1")
    estilo.salvar(fig, cfg.DIR_FIG / "01_validacao_sintetica.png")
    pd.DataFrame(linhas).round(4).to_csv(cfg.DIR_TAB / "validacao_sintetica.csv", index=False)
    print(pd.DataFrame(linhas).round(3).to_string(index=False))

    # --- figura 2: persistência máxima x nível de ruído ----------------------
    snrs = np.array([20, 15, 10, 5, 0, -5, -10, -15])
    seno = takens.padronizar(np.sin(2 * np.pi * 3.0 * T))
    reps = 30
    pm = np.zeros((len(snrs), reps))
    pm_ruido = np.zeros(reps)
    for r in range(reps):
        ru = takens.padronizar(ruido_banda(rng, len(T)))
        pm_ruido[r] = topologia.descritores_h1(
            topologia.h1_rips(takens.imersao_takens(ru, cfg.DIM, cfg.ATRASO, cfg.PASSO_NUVEM))[0])["h1_pers_max"]
        for j, snr in enumerate(snrs):
            x = seno + ru * 10 ** (-snr / 20)
            h1 = topologia.h1_rips(takens.imersao_takens(x, cfg.DIM, cfg.ATRASO, cfg.PASSO_NUVEM))[0]
            pm[j, r] = topologia.descritores_h1(h1)["h1_pers_max"]

    fig, ax = plt.subplots(figsize=(6.6, 3.8), constrained_layout=True)
    q1, med, q3 = np.percentile(pm, [25, 50, 75], axis=1)
    ax.fill_between(snrs, q1, q3, color=estilo.AZUL, alpha=0.18, lw=0)
    ax.plot(snrs, med, color=estilo.AZUL, lw=2, marker="o", ms=5, label="senoide + ruído (mediana, IQR)")
    piso = np.median(pm_ruido)
    ax.axhline(piso, color=estilo.MUDO, lw=1)
    ax.text(snrs.max(), piso, f" só ruído (mediana {piso:.2f})", ha="left", va="bottom",
            fontsize=8.5, color=estilo.TINTA_2)
    ax.invert_xaxis()
    ax.set_xlabel("relação sinal-ruído (dB) — ruído cresce para a direita")
    ax.set_ylabel("persistência máxima em H1")
    ax.set_title("O ciclo persistente desaparece gradualmente com o ruído")
    ax.legend(loc="upper right")
    estilo.salvar(fig, cfg.DIR_FIG / "01_persistencia_vs_ruido.png")
    pd.DataFrame({"snr_db": snrs, "pers_max_q1": q1, "pers_max_mediana": med, "pers_max_q3": q3}
                 ).round(4).to_csv(cfg.DIR_TAB / "validacao_ruido.csv", index=False)


if __name__ == "__main__":
    main()
