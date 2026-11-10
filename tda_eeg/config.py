"""Parâmetros do Projeto 5 (EEG e crises epilépticas via Takens).

Todos os parâmetros foram fixados ANTES de olhar para o desempenho dos
classificadores; os valores de demonstração (canal, banda, fs, janela, d, tau)
são os do código-modelo do enunciado (projetos_TDA.pdf, Seção 5.5).
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR_DADOS = RAIZ / "dataset"
DIR_RES = RAIZ / "resultados"
DIR_FIG = DIR_RES / "figuras"
DIR_TAB = DIR_RES / "tabelas"
DIR_CACHE = DIR_RES / "cache"

RESUMO = DIR_DADOS / "chb01-summary.txt"
URL_BASE = "https://physionet.org/files/chbmit/1.0.0/chb01"

# --- Pré-processamento (Fase 2) -------------------------------------------
CANAL = "FP1-F7"          # canal do enunciado
BANDA = (1.0, 30.0)       # passa-faixa em Hz
FS = 64                   # taxa após reamostragem (Hz)

# --- Janelamento -----------------------------------------------------------
JANELA_S = 8.0            # duração da janela (s)
PASSO_S = 4.0             # passo entre janelas (s)

# --- Imersão de Takens (Fase 3) -------------------------------------------
DIM = 3                   # dimensão de imersão d
ATRASO = 4                # atraso tau em amostras (4/64 Hz = 62,5 ms)
PASSO_NUVEM = 4           # subamostragem temporal da nuvem

# --- Topologia (Fase 4) ----------------------------------------------------
MAX_ARESTA = None         # None = filtração até a maior distância observada
PAISAGEM_K = 3            # número de funções de paisagem (landscapes)
PAISAGEM_GRADE = (0.0, 3.0, 121)   # (mín, máx, nº de pontos) da grade

# --- Rótulos das janelas ---------------------------------------------------
# Uma janela é ICTAL quando pelo menos metade dela cai dentro da crise.
FRACAO_ICTAL = 0.5
# Horizonte pré-ictal: janelas que terminam até HORIZONTE_PRE_S antes do início.
HORIZONTE_PRE_S = 600.0
# Exclusão pós-ictal: janelas até EXCLUSAO_POS_S após o fim da crise não entram
# em nenhuma comparação (nem como interictal).
EXCLUSAO_POS_S = 600.0

CLASSES = ["interictal", "pre-ictal", "ictal"]

# --- Reprodutibilidade -----------------------------------------------------
SEMENTE = 20261110


def garantir_pastas():
    for d in (DIR_RES, DIR_FIG, DIR_TAB, DIR_CACHE):
        d.mkdir(parents=True, exist_ok=True)
