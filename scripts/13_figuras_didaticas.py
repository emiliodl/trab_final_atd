"""Figuras didáticas para a apresentação: uma ideia por figura, com a leitura escrita nela.

  D1_como_ler_diagrama.png      do laço à barra e ao ponto do diagrama de persistência
  D2_expectativa_realidade.png  o que a hipótese previa x o que o EEG mostra na crise
  D3_o_que_e_auroc.png          a AUROC explicada com as janelas do estudo de caso
  D4_todas_as_crises.png        cada crise em uma linha: H1 x amplitude
  D5_crises_detectadas.png      cada crise é uma bolinha: detectada ou não
  D6_linha_do_tempo_chb01.png   as 40 h do chb01: o "estado lento" e as 7 crises

D1 a D3 usam sinais sintéticos e o estudo de caso (chb01, 7 gravações). D4 a D6 leem as
tabelas dos scripts 11 e 12 (todas as gravações dos três pacientes).
Saída em resultados/didaticas/.
"""
import _caminho  # noqa: F401

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Polygon
from sklearn.metrics import roc_auc_score

from tda_eeg import config as cfg
from tda_eeg import dados, estilo, extracao
from tda_eeg.takens import imersao_takens
from tda_eeg.topologia import h1_rips

DIR = cfg.DIR_RES / "didaticas"
TAB = cfg.DIR_MULTI / "tabelas"
AZUL, LARANJA, AGUA, VIOLETA = estilo.AZUL, estilo.LARANJA, estilo.AGUA, estilo.VIOLETA
TINTA, TINTA_2, MUDO, GRADE, EIXO = estilo.TINTA, estilo.TINTA_2, estilo.MUDO, estilo.GRADE, estilo.EIXO
COR_PAC = estilo.COR_PACIENTE
TAM = (16, 6.8)                                              # área útil do slide x 1,32
NOTA = dict(fontsize=15, color=TINTA)                        # texto de anotação
FUNDO = dict(boxstyle="round,pad=0.25", facecolor=estilo.SUPERFICIE, edgecolor="none")   # texto sobre linhas
SETA = dict(arrowstyle="-|>", color=TINTA_2, lw=1.3, shrinkA=2, shrinkB=3)


def estilo_didatico():
    estilo.aplicar()
    mpl.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
                         "xtick.labelsize": 13.5, "ytick.labelsize": 13.5, "figure.titlesize": 19.5,
                         "legend.fontsize": 15, "savefig.dpi": 170})


def passo(ax, n, texto):
    """Título de painel com o número do passo em um círculo."""
    ax.set_title(f"      {texto}", loc="left", pad=12)
    ax.text(0.0, 1.0, str(n), transform=ax.transAxes, ha="left", va="bottom", fontsize=15,
            fontweight="bold", color="white",
            bbox=dict(boxstyle="circle,pad=0.28", facecolor=TINTA_2, edgecolor="none"))


def persistencias(h1):
    return h1[:, 1] - h1[:, 0] if len(h1) else np.zeros(0)


# ---------------------------------------------------------------------------
# D1: como ler um diagrama de persistência
# ---------------------------------------------------------------------------

def d1_como_ler():
    # anel com ruído suficiente para aparecerem também laços pequenos (as barras curtas)
    melhor = None
    for semente in range(200):
        rng = np.random.default_rng(semente)
        n = 46
        ang = np.sort(rng.uniform(0, 2 * np.pi, n))
        raio = 1 + rng.normal(0, 0.17, n)
        X = np.column_stack([raio * np.cos(ang), raio * np.sin(ang)])
        h1 = h1_rips(X)[0]
        p = np.sort(persistencias(h1))[::-1]
        curtas = int(np.sum((p[1:] > 0.03) & (p[1:] < 0.25)))
        if len(p) >= 4 and p[0] > 0.8 and 3 <= curtas <= 6 and p[1] < 0.25:
            melhor = (X, h1)
            break
    X, h1 = melhor
    n = len(X)
    h1 = h1[persistencias(h1) > 0.03]
    p = persistencias(h1)
    k = int(np.argmax(p))
    nasc, morte = h1[k]
    curtas = h1[np.arange(len(h1)) != k]
    curtas = curtas[np.argsort(curtas[:, 0])]
    eps = nasc + 0.40 * (morte - nasc)                       # uma escala em que o laço está vivo

    fig, eixos = plt.subplots(1, 3, figsize=TAM, constrained_layout=True,
                              gridspec_kw={"width_ratios": [1, 1.1, 1]})

    # 1. nuvem + bolas + arestas
    ax = eixos[0]
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= eps:
                for m in range(j + 1, n):
                    if D[i, m] <= eps and D[j, m] <= eps:
                        ax.add_patch(Polygon(X[[i, j, m]], closed=True, facecolor=AZUL, alpha=0.035, lw=0))
    for x in X:
        ax.add_patch(Circle(x, eps / 2, facecolor=AZUL, alpha=0.09, lw=0))
    pares = [(i, j) for i in range(n) for j in range(i + 1, n) if D[i, j] <= eps]
    for i, j in pares:
        ax.plot(X[[i, j], 0], X[[i, j], 1], color=AZUL, lw=0.8, alpha=0.5, zorder=2)
    ax.scatter(X[:, 0], X[:, 1], s=30, color=TINTA, zorder=3)
    ax.text(0, 0, "o buraco:\né isto que\n$H_1$ conta", ha="center", va="center", **NOTA)
    topo = X[np.argmax(X[:, 1] + 0.6 * X[:, 0])]
    ax.annotate("cada ponto ganha\numa bola de raio ε/2", topo, xytext=(1.05, 2.02), ha="center", va="bottom",
                arrowprops=SETA, **NOTA)
    i, j = max(pares, key=lambda ij: (X[ij[0], 0] + X[ij[1], 0]) / 2 - (X[ij[0], 1] + X[ij[1], 1]) / 2
               + 0.8 * D[ij[0], ij[1]])
    ax.annotate("pontos a distância ≤ ε\nviram uma aresta", (X[i] + X[j]) / 2, xytext=(0.95, -2.05), ha="center",
                va="top", arrowprops=SETA, **NOTA)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.45, 2.45)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    passo(ax, 1, f"A nuvem, na escala ε = {eps:.2f}".replace(".", ","))

    # 2. código de barras: as curtas embaixo, a longa em cima
    ax = eixos[1]
    m = len(curtas)
    for i, (b, d) in enumerate(curtas):
        ax.plot([b, d], [i, i], color=MUDO, lw=6, solid_capstyle="butt")
    yk = m + 1.2
    ax.plot([nasc, morte], [yk, yk], color=LARANJA, lw=9, solid_capstyle="butt")
    ax.axvline(eps, color=AZUL, lw=1.6, ls=(0, (4, 3)), ymax=0.93)
    ax.text(eps + 0.015, -1.15, "escala do painel 1", color=TINTA_2, fontsize=13.5, va="center", ha="left")
    ax.annotate("nasce:\no laço fecha", (nasc, yk + 0.15), xytext=(nasc, yk + 1.5), ha="center", va="bottom",
                arrowprops=SETA, **NOTA)
    ax.annotate("morre: o buraco\né preenchido", (morte, yk + 0.15), xytext=(morte - 0.02, yk + 1.5), ha="center",
                va="bottom", arrowprops=SETA, **NOTA)
    ax.text((nasc + morte) / 2, yk + 0.55, f"barra longa: persistência = {morte - nasc:.2f}".replace(".", ","),
            ha="center", va="bottom", fontsize=15, fontweight="bold", color=TINTA, bbox=FUNDO, zorder=5)
    ax.annotate("barras curtas: laços\nque somem logo (ruído)", (curtas[-1, 1] + 0.01, m - 1),
                xytext=(max(curtas[:, 1].max() + 0.22, 0.62), m / 2 - 0.5), ha="left", va="center",
                arrowprops=SETA, bbox=FUNDO, zorder=5, **NOTA)
    ax.set_ylim(-1.8, yk + 3.6)
    ax.set_xlim(0, morte * 1.12)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("escala ε (as bolas crescem para a direita)")
    passo(ax, 2, "Uma barra por laço")

    # 3. diagrama
    ax = eixos[2]
    lim = morte * 1.15
    ax.fill_between([0, lim], [0, lim], [0.28, lim + 0.28], color=GRADE, alpha=0.8, lw=0, zorder=0)
    ax.plot([0, lim], [0, lim], color=EIXO, lw=1.4)
    ax.scatter(curtas[:, 0], curtas[:, 1], s=70, color=MUDO, zorder=3, edgecolor=estilo.SUPERFICIE, linewidths=1)
    ax.scatter([nasc], [morte], s=170, color=LARANJA, zorder=4, edgecolor=estilo.SUPERFICIE, linewidths=1.5)
    ax.annotate("", (nasc, nasc), (nasc, morte - 0.04), arrowprops=dict(arrowstyle="<->", color=LARANJA, lw=1.8))
    ax.text(nasc + 0.05, (nasc + morte) / 2 + 0.05, "persistência:\ndistância até\na diagonal", ha="left",
            va="center", fontsize=15, fontweight="bold", color=TINTA)
    ax.text(nasc, morte + 0.06, "a barra longa", ha="center", va="bottom", **NOTA)
    ax.annotate("as barras curtas:\ncoladas na diagonal", curtas[len(curtas) // 2], xytext=(lim * 0.52, lim * 0.16),
                ha="left", va="center", arrowprops=SETA, **NOTA)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("nascimento (onde a barra começa)")
    ax.set_ylabel("morte (onde a barra termina)")
    passo(ax, 3, "Cada barra vira um ponto")

    fig.suptitle("Laço de verdade = barra longa = ponto longe da diagonal")
    estilo.salvar(fig, DIR / "D1_como_ler_diagrama.png")


# ---------------------------------------------------------------------------
# D2: expectativa x realidade
# ---------------------------------------------------------------------------

def d2_expectativa_realidade():
    fs = cfg.FS
    t = np.arange(int(cfg.JANELA_S * fs)) / fs
    rng = np.random.default_rng(cfg.SEMENTE)
    seno = np.sin(2 * np.pi * 3 * t)
    seno = seno + rng.normal(0, seno.std() / 10 ** (15 / 20), len(t))          # SNR de 15 dB

    g = next(g for g in dados.listar_gravacoes() if g.nome == "chb01_03")
    x, _ = dados.carregar_canal(g.caminho)
    ini = 3024.0                                                                # a janela ictal do slide 13
    eeg = x[int(ini * fs): int(ini * fs) + len(t)]

    linhas = [("O que a hipótese previa: um sinal rítmico", seno, AZUL, "sinal sintético: senoide de 3 Hz com ruído"),
              ("O que o EEG mostra durante a crise", eeg, LARANJA, f"EEG real: chb01_03, {cfg.CANAL}, dentro da crise")]
    fig, eixos = plt.subplots(2, 3, figsize=TAM, constrained_layout=True,
                              gridspec_kw={"width_ratios": [2.5, 1, 1]})
    for i, (titulo, sinal, cor, legenda) in enumerate(linhas):
        z = (sinal - sinal.mean()) / sinal.std()
        X = imersao_takens(sinal, cfg.DIM, cfg.ATRASO, cfg.PASSO_NUVEM)
        h1 = h1_rips(X)[0]
        p = persistencias(h1)
        k = int(np.argmax(p))

        ax = eixos[i, 0]
        ax.plot(t, z, color=cor, lw=1.5)
        ax.set_title(titulo, fontsize=18)
        ax.set_ylabel("sinal padronizado")
        ax.set_xlabel("tempo (s)  ·  " + legenda)
        ax.set_xlim(0, 8)

        ax = eixos[i, 1]
        ax.plot(X[:, 0], X[:, 1], color=GRADE, lw=0.8, zorder=1)
        ax.scatter(X[:, 0], X[:, 1], s=22, color=cor, zorder=2)
        ax.set_aspect("equal")
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-3.6, 3.6)
        ax.set_xlabel("x(t)")
        ax.set_ylabel("x(t + τ)")
        ax.set_title("Nuvem: laço, com buraco" if i == 0 else "Nuvem: novelo, sem buraco")
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])

        ax = eixos[i, 2]
        lim = 3.2
        ax.fill_between([0, lim], [0, lim], [0.28, lim + 0.28], color=GRADE, alpha=0.8, lw=0, zorder=0)
        ax.plot([0, lim], [0, lim], color=EIXO, lw=1.4)
        ax.scatter(h1[:, 0], h1[:, 1], s=40, color=cor, zorder=3, edgecolor=estilo.SUPERFICIE, linewidths=0.8)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("nascimento")
        ax.set_ylabel("morte")
        ax.set_title("Diagrama de $H_1$")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_yticks([0, 1, 2, 3])
        valor = f"{p[k]:.2f}".replace(".", ",")
        if i == 0:
            ax.annotate(f"um ponto longe\nda diagonal:\npersistência\n{valor}", h1[k], xytext=(1.55, 0.95),
                        ha="left", va="center", arrowprops=SETA, fontsize=15, fontweight="bold", color=TINTA)
        else:
            ax.annotate(f"todos colados\nna diagonal:\na maior é {valor}", h1[k], xytext=(1.25, 0.55),
                        ha="left", va="center", arrowprops=SETA, fontsize=15, fontweight="bold", color=TINTA)
    fig.suptitle("Expectativa × realidade: a crise muda o sinal, mas não cria um laço")
    estilo.salvar(fig, DIR / "D2_expectativa_realidade.png")


# ---------------------------------------------------------------------------
# D3: o que é AUROC
# ---------------------------------------------------------------------------

def d3_auroc():
    df = extracao.carregar(extracao.rotulo_config())                            # estudo de caso (7 gravações)
    com_crise = df.gravacao.isin(df.loc[df.classe == "ictal", "gravacao"].unique())
    inter = df[(df.classe == "interictal") & com_crise]
    ict = df[df.classe == "ictal"]
    # (coluna, título, rótulo do eixo, limites, posição x do rótulo do histograma em fração do eixo)
    paineis = [("h1_pers_max", "Persistência máxima de $H_1$ (topologia)", "persistência máxima", (0.1, 1.2), 0.47),
               ("log_variancia", "Log-variância (amplitude do sinal)", "log-variância", (3.5, 10.5), 0.012)]
    rng = np.random.default_rng(cfg.SEMENTE)
    fig, eixos = plt.subplots(2, 2, figsize=TAM, constrained_layout=True,
                              gridspec_kw={"width_ratios": [3.0, 1]})
    for (ax, lado), (col, titulo, rot_x, (lo, hi), x_hist) in zip(eixos, paineis):
        a, b = inter[col].to_numpy(float), ict[col].to_numpy(float)
        auc = roc_auc_score(np.r_[np.ones(len(b)), np.zeros(len(a))], np.r_[b, a])
        cont, bordas = np.histogram(a, bins=np.linspace(lo, hi, 61))
        cont = cont / cont.max()
        ax.bar(bordas[:-1], cont, width=np.diff(bordas), align="edge", color=AZUL, alpha=0.55, lw=0)
        y = 1.25 + rng.uniform(-0.09, 0.09, len(b))
        ax.scatter(np.clip(b, lo, hi), y, s=110, color=LARANJA, zorder=3, edgecolor=estilo.SUPERFICIE, linewidths=1.2)
        ax.text(lo + 0.012 * (hi - lo), 1.62, f"{len(b)} janelas de crise (uma bolinha cada)", color=TINTA,
                fontsize=15, va="center", fontweight="bold")
        ax.text(lo + x_hist * (hi - lo), 0.78, f"{len(a):,} janelas interictais\n(histograma)".replace(",", " "),
                color=TINTA, fontsize=15, va="top", fontweight="bold")
        ax.set_ylim(0, 1.78)
        ax.set_xlim(lo, hi)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(rot_x)
        ax.set_title(titulo, fontsize=17.5)
        lado.axis("off")
        lado.text(0.0, 0.92, "AUROC", fontsize=19, color=TINTA_2, va="center")
        lado.text(0.0, 0.63, f"{auc:.2f}".replace(".", ","), fontsize=60, fontweight="bold", color=TINTA, va="center")
        leitura = ("As bolinhas caem\nno meio do histograma.\nNão dá para separar\ncrise de não crise." if auc < 0.8
                   else "As bolinhas ficam\nà direita do histograma.\nSeparação quase\nperfeita.")
        lado.text(0.0, 0.40, leitura, fontsize=15.5, color=TINTA, va="top", linespacing=1.3)
    fig.suptitle("AUROC = chance de uma janela de crise ter valor maior que uma interictal"
                 "   (0,5 = acaso  ·  1,0 = separação perfeita)", fontsize=17.5)
    estilo.salvar(fig, DIR / "D3_o_que_e_auroc.png")


# ---------------------------------------------------------------------------
# D4: todas as crises, uma linha por crise
# ---------------------------------------------------------------------------

def rotulo_crise(nome):
    grav = nome.split("@")[0]
    return "gravação " + grav.split("_")[1]


def d4_todas_as_crises():
    cr = pd.read_csv(TAB / "auroc_por_crise.csv")
    piv = cr.pivot(index="crise", columns="descritor", values="percentil_blocos")
    ordem = cr.drop_duplicates("crise")[["paciente", "crise", "n_janelas_ictais"]].reset_index(drop=True)
    n = len(ordem)
    fig, ax = plt.subplots(figsize=TAM, constrained_layout=True)
    ax.axvspan(25, 75, color=GRADE, alpha=0.75, lw=0, zorder=0)
    y = np.arange(n)[::-1].astype(float)
    for i in range(1, n):
        if ordem.paciente[i] != ordem.paciente[i - 1]:
            y[i:] -= 0.7                                    # um respiro entre pacientes
    h = piv.loc[ordem.crise, "h1_pers_max"].to_numpy()
    v = piv.loc[ordem.crise, "log_variancia"].to_numpy()
    for yi, hi, vi in zip(y, h, v):
        ax.plot([hi, vi], [yi, yi], color=EIXO, lw=1.4, zorder=1)
    tam = 150 if n <= 14 else 95
    ax.scatter(h, y, s=tam, color=AZUL, zorder=3, edgecolor=estilo.SUPERFICIE, linewidths=1.2,
               label="persistência máxima de $H_1$ (topologia)")
    ax.scatter(v, y, s=tam, marker="D", color=LARANJA, zorder=3, edgecolor=estilo.SUPERFICIE, linewidths=1.2,
               label="log-variância (amplitude)")
    # quando a mesma gravação tem mais de uma crise, numera: "gravação 01 (1ª)", "(2ª)", ...
    grav = ordem.crise.str.split("@").str[0]
    vez = grav.groupby(grav).cumcount() + 1
    rotulos = [rotulo_crise(c) + (f" ({k}ª)" if (grav == g).sum() > 1 else "") for c, g, k in zip(ordem.crise, grav, vez)]
    ax.set_yticks(y, rotulos, fontsize=13.5 if n <= 14 else 10.5)
    for pac in ordem.paciente.unique():
        m = (ordem.paciente == pac).to_numpy()
        ax.text(-22, y[m].mean(), pac, ha="right", va="center", fontsize=17 if n <= 14 else 14, fontweight="bold",
                color=TINTA, clip_on=False)
    ax.set_xlim(-2, 103)
    ax.set_ylim(y.min() - 0.8, y.max() + 1.5)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("percentil do trecho da crise entre trechos interictais de mesma duração (100 = mais extremo que todos)")
    ax.text(50, y.max() + 1.4, "faixa cinza: um trecho interictal comum", ha="center", va="top", fontsize=13.5,
            color=TINTA_2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=15.5)
    # o título conta, nos dados, quantas crises cada descritor põe acima do percentil 90
    fig.suptitle(f"As {n} crises, uma por linha. Acima do percentil 90: amplitude em {int((v >= 90).sum())}, "
                 f"topologia em {int((h >= 90).sum())}")
    estilo.salvar(fig, DIR / "D4_todas_as_crises.png")
    return ordem.assign(h1=h, amplitude=v)


# ---------------------------------------------------------------------------
# D5: crises detectadas, uma bolinha por crise
# ---------------------------------------------------------------------------

def bolinhas(ax, linhas, crises, tam=330):
    """Desenha uma bolinha por crise para cada linha de `linhas` = [(rótulo, detectadas[bool], texto à direita)]."""
    pac = [c[:5] for c in crises]
    x, pos = [], 0.0
    for i, q in enumerate(pac):
        if i and q != pac[i - 1]:
            pos += 0.9
        x.append(pos)
        pos += 1.0
    x = np.array(x)
    for j, (rot, ok, nota) in enumerate(linhas):
        yj = len(linhas) - 1 - j
        for xi, q, d in zip(x, pac, ok):
            ax.scatter(xi, yj, s=tam, marker="o", facecolor=COR_PAC[q] if d else "none", edgecolor=COR_PAC[q],
                       linewidths=2.2, zorder=3)
            if not d:                                           # crise perdida: bolinha vazia com um xis
                ax.plot([xi - 0.15, xi + 0.15], [yj - 0.15, yj + 0.15], color=COR_PAC[q], lw=2, zorder=4)
                ax.plot([xi - 0.15, xi + 0.15], [yj + 0.15, yj - 0.15], color=COR_PAC[q], lw=2, zorder=4)
        ax.text(x.max() + 1.0, yj + 0.13, f"{int(np.sum(ok))} de {len(ok)}", ha="left", va="center", fontsize=24,
                fontweight="bold", color=TINTA)
        ax.text(x.max() + 1.0, yj - 0.27, nota, ha="left", va="center", fontsize=12.5, color=TINTA_2)
        ax.text(x.min() - 0.9, yj, rot, ha="right", va="center", fontsize=16, color=TINTA)
    for q in dict.fromkeys(pac):
        m = np.array([r == q for r in pac])
        ax.text(x[m].mean(), len(linhas) - 0.4, f"{q}\n{m.sum()} crise" + ("s" if m.sum() > 1 else ""), ha="center",
                va="bottom", fontsize=13.5, fontweight="bold", color=TINTA)
    ax.set_ylim(-0.8, len(linhas) + 0.6)
    ax.axis("off")
    return x


def nota_alarmes(fa):
    """Texto dos alarmes falsos por hora, só para os pacientes em que dá para medir."""
    partes = []
    for q, v in fa.items():
        if np.isfinite(v):
            partes.append(f"{q[3:]}: " + (f"{v:.0f}" if v >= 1 or v < 0.05 else f"{v:.1f}".replace(".", ",")))
    return "alarmes falsos/h — " + " · ".join(partes)


def d5_crises_detectadas(fpr=0.01):
    det = pd.read_csv(TAB / "deteccao_por_crise.csv")
    det = det[(det.esquema == "entre pacientes") & np.isclose(det.fpr_treino, fpr)]
    tab = pd.read_csv(TAB / "deteccao_supervisionada.csv")
    tab = tab[(tab.esquema == "entre pacientes") & np.isclose(tab.fpr_treino, fpr)]
    modelos = [("Topológico H1", "Só topologia ($H_1$)"),
               ("Clássico: só forma", "Clássico sem amplitude"),
               ("Clássico: forma + amplitude", "Clássico com amplitude"),
               ("Topológico + clássico", "Topologia + clássico")]
    crises = det.drop_duplicates("crise").crise.tolist()
    linhas = []
    for chave, rot in modelos:
        ok = det[det.descritores == chave].set_index("crise").loc[crises].detectada.to_numpy(bool)
        fa = tab[tab.descritores == chave].set_index("paciente_teste").fa_h
        linhas.append((rot, ok, nota_alarmes(fa)))
    fig, ax = plt.subplots(figsize=TAM, constrained_layout=True)
    x = bolinhas(ax, linhas, crises)
    ax.set_xlim(x.min() - 8.0, x.max() + 10.5)
    ax.text(x.min() - 0.3, -0.74, "bolinha cheia = crise detectada   ·   bolinha com × = crise perdida   ·   "
            "alarmes falsos medidos nos pacientes com interictal a 1 h das crises (01 a 04)", fontsize=12.5,
            color=TINTA_2, va="center")
    n_pac = len(dict.fromkeys(c[:5] for c in crises))
    fig.suptitle(f"Treino em {n_pac - 1} pacientes, teste no que ficou de fora: cada bolinha é uma crise")
    estilo.salvar(fig, DIR / "D5_crises_detectadas.png")


# ---------------------------------------------------------------------------
# D6: as 40 h do chb01
# ---------------------------------------------------------------------------

LIMIAR_ESTADO = 3.3        # persistência total (mediana móvel de 5 min) acima disto = "estado lento"


def estado_lento(df, gravs):
    """Mediana móvel de 5 min da persistência total, por gravação, e a máscara do estado."""
    partes = []
    for g in gravs:
        sub = df[df.gravacao == g.nome].sort_values("t_ini")
        partes.append(sub.assign(pt5=sub.h1_pers_total.rolling(75, center=True, min_periods=20).median(),
                                 lv5=sub.log_variancia.rolling(75, center=True, min_periods=20).median()))
    d = pd.concat(partes)
    d["estado"] = d.pt5 > LIMIAR_ESTADO
    return d


def d6_linha_do_tempo_chb01():
    df = extracao.carregar(f"chb01_{extracao.rotulo_config()}")
    gravs = dados.listar_gravacoes(pacientes=("chb01",), restringir=None)
    d = estado_lento(df, gravs)
    crises = [g.inicio_abs + a for g in gravs for a, _ in g.crises]
    # em que fração do tempo o paciente está no estado, e quantas crises caem nele
    frac = float(d.estado.mean())
    no_estado = 0
    for c in crises:
        perto = d[(d.t_abs + cfg.JANELA_S / 2 - c).abs() < 300]
        no_estado += int(perto.estado.mean() > 0.5)
    resumo = {"limiar": LIMIAR_ESTADO, "fracao_do_tempo_no_estado": frac, "crises": len(crises),
              "crises_no_estado": no_estado, "prob_acaso": frac ** no_estado}
    pd.Series(resumo).to_json(TAB / "estado_lento_chb01.json", indent=1)

    # a primeira gravação começa às 11:42:54 (chb01-summary.txt): relógio = 11,715 h + horas decorridas
    h0 = 11 + 42 / 60 + 54 / 3600
    fig, eixos = plt.subplots(2, 1, figsize=TAM, sharex=True, constrained_layout=True)
    paineis = [("pt5", "Persistência total de $H_1$ (topologia): alta nas faixas salmão"),
               ("lv5", "Log-variância (amplitude): baixa nas mesmas faixas")]
    for ax, (col, titulo) in zip(eixos, paineis):
        for g in gravs:
            sub = d[d.gravacao == g.nome]
            hrs = (sub.t_abs + cfg.JANELA_S / 2).to_numpy() / 3600
            ax.plot(hrs, sub[col], color=AZUL, lw=1.6, zorder=2)
            # faixas do estado: trechos contíguos em que a mediana móvel passa do limiar
            e = sub.estado.to_numpy()
            bordas = np.flatnonzero(np.diff(np.r_[0, e.astype(int), 0]))
            for a_, b_ in zip(bordas[::2], bordas[1::2]):
                if b_ - a_ >= 150:                       # só desenha trechos de 10 min ou mais
                    ax.axvspan(hrs[a_], hrs[min(b_, len(hrs)) - 1], color=estilo.FAIXA_CRISE, lw=0, zorder=0)
        for c in crises:
            ax.axvline(c / 3600, color=LARANJA, lw=2.4, zorder=3)
        ax.set_title(titulo, pad=22 if col == "pt5" else 6)
        ax.grid(axis="x", visible=False)
    for k, c in enumerate(crises, 1):
        eixos[0].text(c / 3600, 1.02, str(k), transform=eixos[0].get_xaxis_transform(), ha="center", va="bottom",
                      fontsize=14.5, fontweight="bold", color=TINTA)
    eixos[0].axhline(LIMIAR_ESTADO, color=TINTA_2, lw=1, ls=(0, (4, 3)), zorder=1)
    ticks = np.arange(np.ceil(h0 / 6) * 6, h0 + d.t_abs.max() / 3600, 6)            # de 6 em 6 h de relógio
    eixos[1].set_xticks(ticks - h0, [f"{int(t % 24):02d}h" for t in ticks])
    eixos[1].set_xlabel("hora do dia  ·  linhas laranja numeradas = as 7 crises  ·  "
                        "faixas salmão = persistência total acima da linha tracejada por 10 min ou mais")
    eixos[1].set_xlim(-0.5, d.t_abs.max() / 3600 + 0.5)
    fig.suptitle(f"As 40 horas do chb01: as {no_estado} crises caem no mesmo estado, "
                 f"que ocupa {100 * frac:.0f} % do tempo (noites e início da tarde)")
    estilo.salvar(fig, DIR / "D6_linha_do_tempo_chb01.png")
    print("  estado lento:", {k: round(v, 5) for k, v in resumo.items()})


def main():
    estilo_didatico()
    DIR.mkdir(parents=True, exist_ok=True)
    import sys
    quais = [a for a in sys.argv[1:] if a.startswith("D")] or ["D1", "D2", "D3", "D4", "D5", "D6"]
    if "D1" in quais:
        d1_como_ler()
    if "D2" in quais:
        d2_expectativa_realidade()
    if "D3" in quais:
        d3_auroc()
    if "D4" in quais:
        print(d4_todas_as_crises().round(0).to_string(index=False))
    if "D5" in quais:
        d5_crises_detectadas()
    if "D6" in quais:
        d6_linha_do_tempo_chb01()


if __name__ == "__main__":
    main()
