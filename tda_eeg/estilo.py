"""Estilo único para todas as figuras (paleta validada para daltonismo)."""
import matplotlib as mpl
import matplotlib.pyplot as plt

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
MUDO = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"

AZUL, LARANJA, AGUA = "#2a78d6", "#eb6834", "#1baf7a"
AMARELO, VIOLETA = "#eda100", "#6250d6"

# a cor segue a classe em todas as figuras
COR_CLASSE = {
    "interictal": AZUL,
    "pre-ictal": AGUA,
    "ictal": LARANJA,
    "pos-ictal": MUDO,
    "transicao": MUDO,
}
ROTULO_CLASSE = {
    "interictal": "interictal",
    "pre-ictal": "pré-ictal",
    "ictal": "ictal",
    "pos-ictal": "pós-ictal",
    "transicao": "transição",
}
FAIXA_CRISE = "#fbe3d8"     # fundo que marca o intervalo anotado da crise

# rampa sequencial (um matiz) e divergente (azul <-> vermelho, meio cinza)
SEQ = mpl.colors.LinearSegmentedColormap.from_list(
    "seq_azul", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = mpl.colors.LinearSegmentedColormap.from_list(
    "div_azul_vermelho", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a9a", "#b72d2d"])


def aplicar():
    mpl.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "text.color": TINTA,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9.5,
        "axes.labelcolor": TINTA_2,
        "axes.edgecolor": EIXO,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRADE,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": MUDO,
        "ytick.color": MUDO,
        "xtick.labelcolor": TINTA_2,
        "ytick.labelcolor": TINTA_2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.titleweight": "semibold",
    })


def faixa_crise(ax, ini, fim, rotulo=None):
    """Sombreia o intervalo da crise; o rótulo evita depender só da cor."""
    ax.axvspan(ini, fim, color=FAIXA_CRISE, lw=0, zorder=0)
    if rotulo:
        ax.text((ini + fim) / 2, 1.0, rotulo, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8.5, color=TINTA_2)


def salvar(fig, caminho):
    fig.savefig(caminho)
    plt.close(fig)
    print(f"  figura: {caminho.name}")
