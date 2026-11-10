"""Animações que mostram o algoritmo em execução (GIF em resultados/animacoes).

1) animacao_filtracao_rips.gif — como nasce um diagrama de persistência: as bolas
   crescem em torno dos pontos da nuvem de Takens, surgem arestas e triângulos
   (complexo de Vietoris-Rips) e o código de barras de H1 é desenhado junto.
   Usa um sinal sintético periódico com ruído, d=2 e poucos pontos, para que o
   complexo caiba no plano e seja legível. Tudo que aparece é calculado de verdade.

2) animacao_janela_deslizante.gif — o pipeline no EEG real (chb01_03): a janela
   de 8 s percorre o sinal em torno da crise; para cada posição mostram-se a
   nuvem de Takens (d=3), o diagrama de H1 e a persistência máxima acumulada.
"""
import _caminho  # noqa: F401
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import EllipseCollection, LineCollection, PolyCollection
from matplotlib.patches import Rectangle
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.distance import pdist, squareform

from tda_eeg import config as cfg
from tda_eeg import dados, estilo, extracao, takens, topologia

DIR_ANIM = cfg.DIR_RES / "animacoes"


# ---------------------------------------------------------------------------
# 1) Filtração de Vietoris-Rips
# ---------------------------------------------------------------------------

def cena_sintetica():
    rng = np.random.default_rng(cfg.SEMENTE)
    fs = cfg.FS
    t = np.arange(int(cfg.JANELA_S * fs)) / fs
    sos = butter(4, cfg.BANDA, btype="band", fs=fs, output="sos")
    ruido = sosfiltfilt(sos, rng.standard_normal(len(t) + 4 * fs))[2 * fs: 2 * fs + len(t)]
    x = takens.padronizar(np.sin(2 * np.pi * 3.0 * t)) + takens.padronizar(ruido) * 10 ** (-10 / 20)
    X = takens.imersao_takens(x, dim=2, atraso=5, passo=12)
    return t, takens.padronizar(x), X


def animar_filtracao(n_quadros=84, fps=10):
    t, x, X = cena_sintetica()
    n = len(X)
    D = squareform(pdist(X))
    h1, _, _ = topologia.h1_rips(X)
    h1 = h1[topologia.persistencias(h1) > 1e-9]
    h1 = h1[np.argsort(h1[:, 0])]
    principal = int(np.argmax(topologia.persistencias(h1)))

    iu = np.triu_indices(n, 1)
    arestas = np.column_stack(iu)
    val_aresta = D[iu]
    tri = np.array(list(combinations(range(n), 3)))
    val_tri = np.max([D[tri[:, 0], tri[:, 1]], D[tri[:, 0], tri[:, 2]], D[tri[:, 1], tri[:, 2]]], axis=0)

    eps_max = min(h1[principal, 1] * 1.12, D.max())
    grade = np.r_[np.linspace(0, eps_max, n_quadros), np.full(14, eps_max)]   # pausa no fim
    betti = np.array([np.sum((h1[:, 0] <= e) & (h1[:, 1] > e)) for e in grade])

    fig = plt.figure(figsize=(11, 6.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.9], width_ratios=[1.2, 1])
    ax_s = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[1, 0])
    gd = gs[:, 1].subgridspec(2, 1, height_ratios=[1.5, 1])
    ax_b = fig.add_subplot(gd[0])
    ax_n = fig.add_subplot(gd[1], sharex=ax_b)

    ax_s.plot(t, x, color=estilo.AZUL, lw=1)
    ax_s.set_title("1. Sinal: senoide de 3 Hz com ruído (SNR 10 dB)")
    ax_s.set_xlabel("tempo (s)")

    lim = np.abs(X).max() + eps_max / 2 * 0.55
    ax_c.set_xlim(-lim, lim)
    ax_c.set_ylim(-lim, lim)
    ax_c.set_aspect("equal")
    ax_c.set_xlabel("x(t)")
    ax_c.set_ylabel("x(t + τ)")
    ax_c.grid(False)
    ax_c.scatter(X[:, 0], X[:, 1], s=16, color=estilo.TINTA, zorder=5, linewidths=0)
    ax_c.set_title("2. Nuvem de Takens e complexo de Vietoris-Rips")
    titulo_c = ax_c.text(0.02, 0.985, "", transform=ax_c.transAxes, ha="left", va="top",
                         fontsize=9.5, color=estilo.TINTA_2)

    ax_b.set_xlim(0, eps_max)
    ax_b.set_ylim(-1, len(h1))
    ax_b.set_yticks([])
    ax_b.grid(axis="y", visible=False)
    ax_b.set_title("3. Código de barras de H1 (cada barra é um ciclo)")
    ax_b.tick_params(labelbottom=False)
    rot = ax_b.text(0, 0, "", fontsize=9, color=estilo.TINTA_2, va="center")
    ax_n.plot(grade[:n_quadros], betti[:n_quadros], color=estilo.GRADE, lw=1.5, drawstyle="steps-post")
    curva, = ax_n.plot([], [], color=estilo.AZUL, lw=2, drawstyle="steps-post")
    ponto, = ax_n.plot([], [], "o", color=estilo.AZUL, ms=6)
    ax_n.set_ylim(-0.3, betti.max() + 0.7)
    ax_n.set_xlabel("escala ε (distância máxima entre pontos ligados)")
    ax_n.set_ylabel("β1")
    ax_n.set_title("Número de ciclos vivos em cada escala (curva de Betti)")
    linha_b = ax_b.axvline(0, color=estilo.TINTA_2, lw=1)
    linha_n = ax_n.axvline(0, color=estilo.TINTA_2, lw=1)
    fig.suptitle("Do sinal ao diagrama de persistência: filtração de Vietoris-Rips na nuvem de Takens")

    estado = {"art": []}

    def quadro(k):
        e = grade[k]
        for a in estado["art"]:
            a.remove()
        bolas = EllipseCollection(np.full(n, e), np.full(n, e), np.zeros(n), units="xy",
                                  offsets=X, offset_transform=ax_c.transData,
                                  facecolor=estilo.AZUL, alpha=0.10, edgecolor="none", zorder=1)
        tris = PolyCollection(X[tri[val_tri <= e]], facecolor=estilo.AZUL, alpha=0.16,
                              edgecolor="none", zorder=2)
        segs = LineCollection(X[arestas[val_aresta <= e]], colors=estilo.AZUL, linewidths=0.8,
                              alpha=0.75, zorder=3)
        vivos = h1[:, 0] <= e
        barras = LineCollection(
            [[(b, i), (min(d, e), i)] for i, (b, d) in enumerate(h1) if vivos[i]],
            colors=[estilo.LARANJA if i == principal else estilo.AZUL
                    for i in range(len(h1)) if vivos[i]],
            linewidths=4, capstyle="butt")
        for ax, col in ((ax_c, bolas), (ax_c, tris), (ax_c, segs), (ax_b, barras)):
            ax.add_collection(col)
        estado["art"] = [bolas, tris, segs, barras]
        titulo_c.set_text(f"ε = {e:.2f}  ·  {int((val_aresta <= e).sum())} arestas  ·  "
                          f"{int((val_tri <= e).sum())} triângulos  ·  β1 = {betti[k]}")
        if vivos[principal]:
            rot.set_position((h1[principal, 0], principal + 0.32))
            rot.set_text("ciclo principal: o laço do sinal periódico")
        curva.set_data(grade[: k + 1], betti[: k + 1])
        ponto.set_data([e], [betti[k]])
        linha_b.set_xdata([e, e])
        linha_n.set_xdata([e, e])
        return []

    anim = FuncAnimation(fig, quadro, frames=len(grade), blit=False)
    anim.save(DIR_ANIM / "animacao_filtracao_rips.gif", writer=PillowWriter(fps=fps), dpi=84)

    # quadros-chave em PNG (para relatório/slides, onde o GIF não anima)
    chaves = [0.12, 0.3, 0.62, 1.0]
    for j, f in enumerate(chaves, 1):
        quadro(int(f * (n_quadros - 1)))
        fig.savefig(cfg.DIR_FIG / f"09_filtracao_quadro{j}.png", dpi=130)
    plt.close(fig)
    print(f"  animacao_filtracao_rips.gif ({len(grade)} quadros, {n} pontos, "
          f"persistência do ciclo principal {topologia.persistencias(h1)[principal]:.2f})")


# ---------------------------------------------------------------------------
# 2) Janela deslizante no EEG real
# ---------------------------------------------------------------------------

def animar_janela(antes=60.0, depois=170.0, passo=2, fps=10):
    gravs = dados.listar_gravacoes()
    g = next(g for g in gravs if g.crises)
    ini, fim = g.crises[0]
    df, diags = extracao.carregar("zoom", com_diagramas=True)
    m = ((df.gravacao == g.nome) & (df.t_ini >= ini - antes) & (df.t_fim <= ini + depois)).to_numpy()
    idx = np.flatnonzero(m)[::passo]
    x, fs = dados.carregar_canal(g.caminho)
    w = int(cfg.JANELA_S * fs)

    t_ini = df.t_ini.to_numpy()
    centro = t_ini[idx] + cfg.JANELA_S / 2 - ini
    pmax = df.h1_pers_max.to_numpy()[idx]
    # limite do diagrama: morte do ciclo mais persistente entre todos os quadros
    # (pontos de ruído colados à diagonal em escalas maiores ficam fora do quadro)
    lim_d = max(2.5, 1.15 * max(
        diags[i][np.argmax(topologia.persistencias(diags[i])), 1] for i in idx))
    rot_classe = {"interictal": "antes da crise", "pre-ictal": "pré-ictal", "ictal": "ICTAL",
                  "transicao": "transição", "pos-ictal": "pós-ictal"}

    fig = plt.figure(figsize=(11.5, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.55])
    ax_e = fig.add_subplot(gs[0, :])
    ax_c, ax_d, ax_p = (fig.add_subplot(gs[1, j]) for j in range(3))

    i0, i1 = int((ini - antes) * fs), int((ini + depois) * fs)
    ax_e.plot(np.arange(i0, i1) / fs - ini, x[i0:i1], color=estilo.TINTA_2, lw=0.5)
    estilo.faixa_crise(ax_e, 0, fim - ini, "crise anotada")
    ax_e.set_xlim(-antes, depois)
    ax_e.set_ylabel("EEG (µV)")
    ax_e.set_xlabel("tempo relativo ao início da crise (s)")
    titulo_e = ax_e.set_title("")
    yl = ax_e.get_ylim()
    janela = Rectangle((0, yl[0]), cfg.JANELA_S, yl[1] - yl[0], facecolor=estilo.AZUL, alpha=0.22,
                       edgecolor=estilo.AZUL, lw=1.2, zorder=4)
    ax_e.add_patch(janela)

    ax_c.set_xlim(-4.2, 4.2)
    ax_c.set_ylim(-4.2, 4.2)
    ax_c.set_aspect("equal")
    ax_c.set_xlabel("x(t)")
    ax_c.set_ylabel("x(t + τ)")
    ax_c.set_title(f"Nuvem de Takens (d={cfg.DIM}, τ={cfg.ATRASO})")
    traj, = ax_c.plot([], [], color=estilo.GRADE, lw=0.7, zorder=1)
    nuvem = ax_c.scatter([], [], s=13, linewidths=0, zorder=2)

    ax_d.plot([0, lim_d], [0, lim_d], color=estilo.EIXO, lw=0.9)
    ax_d.set_xlim(0, lim_d)
    ax_d.set_ylim(0, lim_d)
    ax_d.set_aspect("equal")
    ax_d.set_xlabel("nascimento")
    ax_d.set_ylabel("morte")
    titulo_d = ax_d.set_title("")
    diag = ax_d.scatter([], [], s=20, linewidths=0, zorder=3)

    estilo.faixa_crise(ax_p, 0, fim - ini)
    ax_p.plot(centro, pmax, color=estilo.GRADE, lw=1.2)
    tr, = ax_p.plot([], [], color=estilo.AZUL, lw=1.8)
    pt, = ax_p.plot([], [], "o", color=estilo.AZUL, ms=6)
    ax_p.set_xlim(-antes, depois)
    ax_p.set_ylim(0, pmax.max() * 1.08)
    ax_p.set_xlabel("centro da janela (s)")
    ax_p.set_title("Persistência máxima de H1")
    fig.suptitle(f"O pipeline no EEG real ({g.nome}, {cfg.CANAL}): janela → Takens → Rips → H1")

    def quadro(k):
        i = idx[k]
        seg = x[int(round(t_ini[i] * fs)): int(round(t_ini[i] * fs)) + w]
        X = takens.imersao_takens(seg, cfg.DIM, cfg.ATRASO, cfg.PASSO_NUVEM)
        classe = df.classe.iloc[i]
        cor = estilo.COR_CLASSE.get(classe, estilo.AZUL)
        janela.set_x(t_ini[i] - ini)
        janela.set_facecolor(cor)
        janela.set_edgecolor(cor)
        titulo_e.set_text(f"Janela de 8 s em {t_ini[i] - ini:+.0f} s — estado: {rot_classe[classe]}")
        traj.set_data(X[:, 0], X[:, 1])
        nuvem.set_offsets(X[:, :2])
        nuvem.set_color(cor)
        h1 = diags[i]
        diag.set_offsets(h1)
        diag.set_color(cor)
        titulo_d.set_text(f"Diagrama de H1 — {len(h1)} classes")
        tr.set_data(centro[: k + 1], pmax[: k + 1])
        pt.set_data([centro[k]], [pmax[k]])
        return []

    anim = FuncAnimation(fig, quadro, frames=len(idx), blit=False)
    anim.save(DIR_ANIM / "animacao_janela_deslizante.gif", writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    print(f"  animacao_janela_deslizante.gif ({len(idx)} quadros)")


def main():
    cfg.garantir_pastas()
    DIR_ANIM.mkdir(parents=True, exist_ok=True)
    estilo.aplicar()
    animar_filtracao()
    animar_janela()


if __name__ == "__main__":
    main()
