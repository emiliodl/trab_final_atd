"""Definição supervisionada do enunciado (Seção 6.4): rótulos, saídas e avaliação.

Usa os descritores já extraídos pelo script 10 (nada é recalculado) e segue a
versão recomendada da tarefa:

Rótulos de DETECÇÃO (binária)
  y = 1  janela TOTALMENTE contida em uma crise anotada;
  y = 0  janela interictal a pelo menos AFASTAMENTO_S (1 h) de qualquer crise;
  sem rótulo  todo o resto (janelas que cruzam início/fim, pré e pós-ictal, e o
         interictal próximo das crises). Não entram no treino nem nas métricas
         por janela, mas recebem escore para a avaliação por evento.

Extensão de TRÊS CLASSES
  pré-ictal  janela contida em [ts - T_PRE, ts - T_EXCL), com T_PRE = 600 s e
             T_EXCL = 0 s (os mesmos valores fixados no config da análise original);
  ictal e interictal como acima; o resto fica sem rótulo.
  Saída: vetor de 3 probabilidades e classe prevista por janela.

Avaliação (hiperparâmetros fixos, partições por gravação: sem janelas sobrepostas
entre treino e teste)
  * entre pacientes: deixa-um-paciente-fora;
  * personalizada cronológica: em cada paciente, treino nas gravações até a que
    contém a crise floor(n/2) e teste nas gravações POSTERIORES.
  Métricas por janela (AUROC, AUPRC, prevalência) e por evento, com um limiar FIXADO
  NO TREINO (1 % de falsos positivos por janela entre os negativos de treino):
  sensibilidade por crise, atraso de detecção e alarmes falsos por hora.

Saída em resultados/multipaciente/{tabelas,figuras}.
"""
import _caminho  # noqa: F401
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tda_eeg import avaliacao, baseline, estilo, extracao, topologia
from tda_eeg import config as cfg
from tda_eeg import dados

DIR_FIG = cfg.DIR_MULTI / "figuras"
DIR_TAB = cfg.DIR_MULTI / "tabelas"

AFASTAMENTO_S = 3600.0          # interictal "suficientemente afastado"
T_PRE, T_EXCL = cfg.HORIZONTE_PRE_S, 0.0
# Dois pontos de operação, ambos fixados no treino: limiar = quantil (1 - fpr) do
# escore das janelas negativas de treino. 1 % por janela equivale a ~9 janelas
# positivas por hora; 0,1 % a ~1 por hora.
FPRS_TREINO = (0.01, 0.001)
CONJUNTOS = {
    "Topológico H1": topologia.COLUNAS_TOPO,
    "Clássico: só forma": baseline.COLUNAS_FORMA,
    "Clássico: forma + amplitude": baseline.COLUNAS_BASE,
    "Topológico + clássico": topologia.COLUNAS_TOPO + baseline.COLUNAS_BASE,
}
CLASSES3 = ["interictal", "pre-ictal", "ictal"]


def modelo():
    return avaliacao.modelos()["Regressão logística"]


def modelo3():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))


# ---------------------------------------------------------------------------
# Dados e rótulos
# ---------------------------------------------------------------------------

def carregar(gravs):
    partes = []
    for pac in cfg.PACIENTES_MULTI:
        d = extracao.carregar(f"{pac}_{extracao.rotulo_config()}")
        d.insert(0, "paciente", pac)
        partes.append(d)
    df = pd.concat(partes, ignore_index=True)
    ordem = {g.nome: i for i, g in enumerate(gravs)}       # ordem cronológica por paciente
    df["ordem_grav"] = df.gravacao.map(ordem)
    return df.sort_values(["paciente", "t_abs"]).reset_index(drop=True)


def crises_por_paciente(gravs):
    """{paciente: [(nome_crise, gravacao, ts_abs, te_abs), ...]} em ordem cronológica."""
    out = {}
    for g in gravs:
        for a, b in g.crises:
            out.setdefault(g.paciente, []).append(
                (f"{g.nome}@{int(a)}s", g.nome, g.inicio_abs + a, g.inicio_abs + b))
    return {p: sorted(v, key=lambda c: c[2]) for p, v in out.items()}


def rotular(df, crises):
    a = df.t_abs.to_numpy()
    b = a + cfg.JANELA_S
    y = np.full(len(df), np.nan)
    c3 = np.full(len(df), None, dtype=object)
    for pac, cs in crises.items():
        m = (df.paciente == pac).to_numpy()
        dist = np.full(len(df), np.inf)
        dentro = np.zeros(len(df), bool)
        pre = np.zeros(len(df), bool)
        for _, _, ts, te in cs:
            dist = np.minimum(dist, np.maximum.reduce([ts - b, a - te, np.zeros(len(df))]))
            dentro |= (a >= ts) & (b <= te)
            pre |= (a >= ts - T_PRE) & (b <= ts - T_EXCL)
        longe = dist >= AFASTAMENTO_S
        y[m & dentro] = 1
        y[m & longe] = 0
        c3[m & longe] = "interictal"
        c3[m & pre & ~dentro] = "pre-ictal"
        c3[m & dentro] = "ictal"
    df = df.copy()
    df["y_det"] = y
    df["classe3"] = c3
    return df


# ---------------------------------------------------------------------------
# Partições
# ---------------------------------------------------------------------------

def particoes(df, crises):
    """(esquema, paciente testado, máscara de treino, máscara de teste)."""
    out = []
    for pac in cfg.PACIENTES_MULTI:
        teste = (df.paciente == pac).to_numpy()
        out.append(("entre pacientes", pac, ~teste, teste))
    for pac, cs in crises.items():
        k = len(cs) // 2                                  # floor(n/2) crises no treino
        m = (df.paciente == pac).to_numpy()
        if k == 0:                                        # uma crise só: não há o que treinar antes dela
            out.append(("cronológica (mesmo paciente)", pac, np.zeros(len(df), bool), m))
            continue
        ultima = df.loc[df.gravacao == cs[k - 1][1], "ordem_grav"].iloc[0]
        treino = m & (df.ordem_grav <= ultima).to_numpy()
        teste = m & (df.ordem_grav > ultima).to_numpy()
        out.append(("cronológica (mesmo paciente)", pac, treino, teste))
    return out


# ---------------------------------------------------------------------------
# Métricas por evento
# ---------------------------------------------------------------------------

def eventos(df, teste, s, limiar, crises_pac):
    """Sensibilidade por crise, atraso de detecção e alarmes falsos/h no teste."""
    sub = df[teste]
    st = s[teste]
    alarme = st >= limiar
    neg = (sub.y_det == 0).to_numpy()
    horas = neg.sum() * cfg.PASSO_S / 3600
    falsos = sum(avaliacao.sequencias(alarme[m] & neg[m])
                 for m in [(sub.gravacao == g).to_numpy() for g in sub.gravacao.unique()])
    a = sub.t_abs.to_numpy()
    b = a + cfg.JANELA_S
    detalhes = []
    for nome, grav, ts, te in crises_pac:
        if grav not in set(sub.gravacao):
            continue
        sobre = (a < te) & (b > ts)                      # janelas que tocam a crise
        hit = np.flatnonzero(sobre & alarme)
        detalhes.append({"crise": nome, "detectada": len(hit) > 0,
                         "atraso_s": float(b[hit[0]] - ts) if len(hit) else np.nan})
    return {
        "crises_teste": len(detalhes),
        "crises_detectadas": sum(d["detectada"] for d in detalhes),
        "atraso_mediano_s": float(np.nanmedian([d["atraso_s"] for d in detalhes]))
        if any(d["detectada"] for d in detalhes) else np.nan,
        "fa_h": falsos / horas if horas else np.nan,
        "horas_interictais_teste": horas,
    }, detalhes


# ---------------------------------------------------------------------------
# Detecção binária
# ---------------------------------------------------------------------------

def deteccao(df, crises):
    y = df.y_det.to_numpy()
    rot = np.isfinite(y)
    linhas, por_crise, saidas = [], [], []
    for esquema, pac, treino, teste in particoes(df, crises):
        tr = treino & rot
        te = teste & rot
        base = {"esquema": esquema, "paciente_teste": pac,
                "ictais_treino": int((y[tr] == 1).sum()), "interictais_treino": int((y[tr] == 0).sum()),
                "ictais_teste": int((y[te] == 1).sum()), "interictais_teste": int((y[te] == 0).sum())}
        if base["ictais_treino"] == 0 or base["interictais_treino"] == 0 or base["ictais_teste"] == 0:
            # sem as duas classes no treino não há modelo, e sem crise no teste não há o que detectar:
            # registrado, não contornado
            motivo = ("teste sem crises (não há gravações posteriores)" if base["ictais_teste"] == 0
                      else "treino sem janelas das duas classes")
            for nome_c in CONJUNTOS:
                linhas.append({**base, "descritores": nome_c, "obs": "não avaliável: " + motivo})
            print(f"  {esquema:30s} {pac}: não avaliável ({motivo}; treino com "
                  f"{base['ictais_treino']} ictais e {base['interictais_treino']} interictais)")
            continue
        for nome_c, cols in CONJUNTOS.items():
            X = df[cols].to_numpy(float)
            m = clone(modelo()).fit(X[tr], y[tr])
            s = np.full(len(df), np.nan)
            s[teste] = m.predict_proba(X[teste])[:, 1]
            neg_tr = m.predict_proba(X[tr & (y == 0)])[:, 1]
            r0 = {**base, "descritores": nome_c,
                  "prevalencia_teste": float(y[te].mean()) if te.any() else np.nan,
                  "auroc": roc_auc_score(y[te], s[te]) if 0 < y[te].sum() < te.sum() else np.nan,
                  # paciente sem interictal a 1 h das crises: só as métricas por evento fazem sentido
                  "auprc": average_precision_score(y[te], s[te]) if 0 < y[te].sum() < te.sum() else np.nan}
            prev_cols = {}
            for fpr in FPRS_TREINO:
                limiar = float(np.quantile(neg_tr, 1 - fpr))
                ev, det = eventos(df, teste, s, limiar, crises[pac])
                linhas.append({**r0, "fpr_treino": fpr, "limiar": limiar, **ev})
                por_crise += [{"esquema": esquema, "descritores": nome_c, "fpr_treino": fpr, **d}
                              for d in det]
                prev_cols[f"prevista_fpr{fpr:g}"] = (s[teste] >= limiar).astype(int)
            if esquema == "entre pacientes":
                saidas.append(pd.DataFrame({
                    "paciente": df.paciente[teste], "gravacao": df.gravacao[teste],
                    "canal": df.canal[teste], "t_ini": df.t_ini[teste], "t_abs": df.t_abs[teste],
                    "y_det": y[teste], "descritores": nome_c, "escore": s[teste], **prev_cols}))
        print(f"  {esquema:30s} {pac}: " + "  ".join(
            f"{r['descritores'][:14]} [{r['fpr_treino']:g}] AUROC {r['auroc']:.2f} sens "
            f"{r['crises_detectadas']}/{r['crises_teste']} FA/h {r['fa_h']:.1f}"
            for r in linhas[-2 * len(CONJUNTOS):]), flush=True)
    return pd.DataFrame(linhas), pd.DataFrame(por_crise), pd.concat(saidas, ignore_index=True)


# ---------------------------------------------------------------------------
# Controle: o ganho de somar a topologia ao baseline é real?
# ---------------------------------------------------------------------------

def controle_embaralhado(df, n_rep=20):
    """AUPRC entre pacientes do baseline, do baseline + topologia e do baseline + topologia
    EMBARALHADA (as linhas dos descritores de H1 são permutadas dentro de cada paciente).

    Se o ganho viesse só de ter mais 15 colunas, a versão embaralhada ganharia o mesmo.
    """
    y = df.y_det.to_numpy()
    rot = np.isfinite(y)
    XB = df[baseline.COLUNAS_BASE].to_numpy(float)
    XT = df[topologia.COLUNAS_TOPO].to_numpy(float)
    pacs = list(cfg.PACIENTES_MULTI)

    def auprc(X):
        saida = {}
        for pac in pacs:
            teste = (df.paciente == pac).to_numpy()
            m = clone(modelo()).fit(X[~teste & rot], y[~teste & rot])
            tt = teste & rot
            saida[pac] = (average_precision_score(y[tt], m.predict_proba(X[tt])[:, 1])
                          if 0 < y[tt].sum() < tt.sum() else np.nan)      # precisa das duas classes no teste
        return saida

    base, com_topo = auprc(XB), auprc(np.c_[XB, XT])
    rng = np.random.default_rng(cfg.SEMENTE)
    reps = []
    for _ in range(n_rep):
        XS = XT.copy()
        for pac in pacs:
            idx = np.flatnonzero((df.paciente == pac).to_numpy())
            XS[idx] = XT[rng.permutation(idx)]
        reps.append(auprc(np.c_[XB, XS]))
    reps = pd.DataFrame(reps)
    return pd.DataFrame({"paciente_teste": pacs,
                         "auprc_classico": [base[p] for p in pacs],
                         "auprc_classico_mais_topologia": [com_topo[p] for p in pacs],
                         "auprc_topologia_embaralhada_media": [reps[p].mean() for p in pacs],
                         "auprc_topologia_embaralhada_max": [reps[p].max() for p in pacs],
                         "repeticoes": n_rep})


# ---------------------------------------------------------------------------
# Três classes
# ---------------------------------------------------------------------------

def tres_classes(df):
    c3 = df.classe3.to_numpy()
    rot = pd.notna(c3)
    linhas, matrizes, saidas = [], {}, []
    for nome_c, cols in [(k, CONJUNTOS[k]) for k in
                         ("Topológico H1", "Clássico: forma + amplitude", "Topológico + clássico")]:
        X = df[cols].to_numpy(float)
        proba = np.full((len(df), 3), np.nan)
        for pac in cfg.PACIENTES_MULTI:
            teste = (df.paciente == pac).to_numpy()
            tr = ~teste & rot
            m = clone(modelo3()).fit(X[tr], c3[tr])
            ordem = [list(m.classes_).index(c) for c in CLASSES3]
            proba[teste] = m.predict_proba(X[teste])[:, ordem]
        prev = np.array(CLASSES3, dtype=object)[np.nanargmax(np.nan_to_num(proba, nan=-1), axis=1)]
        for pac in cfg.PACIENTES_MULTI:
            mm = rot & (df.paciente == pac).to_numpy()
            r = {"descritores": nome_c, "paciente_teste": pac,
                 "acuracia_balanceada": balanced_accuracy_score(c3[mm], prev[mm])}
            for j, c in enumerate(CLASSES3):
                n_c = int((c3[mm] == c).sum())
                r[f"auroc_{c}_vs_resto"] = roc_auc_score(c3[mm] == c, proba[mm, j]) if 0 < n_c < mm.sum() else np.nan
                r[f"n_{c}"] = n_c
            linhas.append(r)
        matrizes[nome_c] = confusion_matrix(c3[rot].astype(str), prev[rot].astype(str),
                                            labels=CLASSES3).tolist()
        saidas.append(pd.DataFrame({
            "paciente": df.paciente[rot], "gravacao": df.gravacao[rot], "t_ini": df.t_ini[rot],
            "classe3": c3[rot], "descritores": nome_c,
            **{f"p_{c}": proba[rot, j] for j, c in enumerate(CLASSES3)},
            "classe_prevista": prev[rot]}))
        print(f"  3 classes {nome_c:28s} " + "  ".join(
            f"{r['paciente_teste']} bal.acc {r['acuracia_balanceada']:.2f} "
            f"AUROC pré {r['auroc_pre-ictal_vs_resto']:.2f}" for r in linhas[-3:]), flush=True)
    return pd.DataFrame(linhas), matrizes, pd.concat(saidas, ignore_index=True)


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------

def figura(tab):
    nomes = list(CONJUNTOS)
    esquemas = ["entre pacientes", "cronológica (mesmo paciente)"]
    cores, marcas = estilo.COR_PACIENTE, estilo.MARCA_PACIENTE
    n_pac = len(cfg.PACIENTES_MULTI)
    fig, eixos = plt.subplots(2, 2, figsize=(13.5, 8.6), sharey=True, constrained_layout=True)
    yy = np.arange(len(nomes))[::-1]
    tab = tab[tab.fpr_treino.notna()]
    for i, esq in enumerate(esquemas):
        for j, (col, xlab, escala) in enumerate([
                ("sens", "crises detectadas no teste (%)", "linear"),
                ("fa_h", "alarmes falsos por hora (escala log)", "log")]):
            ax = eixos[i, j]
            for k, pac in enumerate(cfg.PACIENTES_MULTI):
                for fpr, cheio in zip(FPRS_TREINO, (False, True)):
                    sub = tab[(tab.esquema == esq) & (tab.paciente_teste == pac)
                              & (tab.fpr_treino == fpr)].set_index("descritores")
                    if sub.empty:
                        continue
                    if col == "sens":
                        v = 100 * sub.crises_detectadas / sub.crises_teste
                    else:
                        v = sub.fa_h.clip(lower=0.05)
                    desloc = (k - (n_pac - 1) / 2) * min(0.2, 0.8 / n_pac) + (0.035 if cheio else -0.035)
                    ax.scatter(v.loc[nomes], yy + desloc, s=50, marker=marcas[pac],
                               facecolor=cores[pac] if cheio else "none", edgecolor=cores[pac],
                               linewidths=1.3, zorder=3,
                               label=f"{pac} · {100 * fpr:g} %" if i == 0 and j == 1 else None)
            ax.set_xscale(escala)
            if col == "sens":
                ax.set_xlim(-5, 105)
            ax.set_xlabel(xlab)
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{esq.capitalize()}: " + ("sensibilidade por crise" if col == "sens"
                                                     else "alarmes falsos/h"), fontsize=10.5)
        eixos[i, 0].set_yticks(yy, nomes)
    eixos[0, 1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5,
                       title="paciente · FP no treino")
    fig.suptitle("Detecção por evento com limiar fixado no treino "
                 "(vazio: 1 % de falsos positivos por janela; cheio: 0,1 %)")
    estilo.salvar(fig, DIR_FIG / "12_deteccao_por_evento.png")


def main():
    estilo.aplicar()
    DIR_FIG.mkdir(parents=True, exist_ok=True)
    DIR_TAB.mkdir(parents=True, exist_ok=True)
    gravs = dados.listar_gravacoes(pacientes=cfg.PACIENTES_MULTI, restringir=None)
    crises = crises_por_paciente(gravs)
    df = rotular(carregar(gravs), crises)

    cont = df.groupby("paciente").agg(
        ictal_total=("y_det", lambda v: int((v == 1).sum())),
        interictal_longe=("y_det", lambda v: int((v == 0).sum())),
        sem_rotulo=("y_det", lambda v: int(v.isna().sum())))
    cont = cont.join(df.pivot_table(index="paciente", columns="classe3", values="t_ini",
                                    aggfunc="size").add_prefix("3c_"))
    cont.to_csv(DIR_TAB / "rotulos_supervisionados.csv")
    print(cont.to_string(), "\n")

    for esq, pac, tr, te in particoes(df, crises):
        if esq.startswith("cron") and tr.any() and te.any():
            print(f"  partição cronológica {pac}: treino {sorted(df.gravacao[tr].unique())[-1]} e "
                  f"anteriores | teste {', '.join(sorted(df.gravacao[te].unique()))}")
    print()

    tab, por_crise, saida = deteccao(df, crises)
    tab.round(4).to_csv(DIR_TAB / "deteccao_supervisionada.csv", index=False)
    por_crise.to_csv(DIR_TAB / "deteccao_por_crise.csv", index=False)
    saida.to_csv(DIR_TAB / "saida_deteccao_por_janela.csv.gz", index=False)
    pd.set_option("display.width", 250)
    print("\n" + tab.drop(columns=["limiar", "horas_interictais_teste"], errors="ignore")
          .round(3).to_string(index=False))

    ctrl = controle_embaralhado(df)
    ctrl.round(4).to_csv(DIR_TAB / "controle_topologia.csv", index=False)
    print("\nControle (AUPRC entre pacientes):\n" + ctrl.round(3).to_string(index=False))

    print()
    t3, matrizes, saida3 = tres_classes(df)
    t3.round(4).to_csv(DIR_TAB / "tres_classes.csv", index=False)
    saida3.to_csv(DIR_TAB / "saida_tres_classes_por_janela.csv.gz", index=False)
    (DIR_TAB / "tres_classes_matrizes.json").write_text(
        json.dumps({"ordem": CLASSES3, "linhas=verdade, colunas=previsto": matrizes},
                   indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n" + t3.round(2).to_string(index=False))
    for k, mtz in matrizes.items():
        print(f"  matriz de confusão (deixa-um-paciente-fora) {k}: {mtz}")

    figura(tab)
    (DIR_TAB / "configuracao_supervisionada.json").write_text(json.dumps({
        "ictal": "janela totalmente contida na crise anotada",
        "interictal": f"janela a pelo menos {AFASTAMENTO_S:.0f} s de qualquer crise",
        "pre_ictal": f"janela contida em [ts - {T_PRE:.0f} s, ts - {T_EXCL:.0f} s)",
        "excluidas": "todas as demais (cruzam início/fim, pós-ictal, interictal próximo)",
        "limiares": [f"quantil {1 - f:g} do escore dos negativos de treino" for f in FPRS_TREINO],
        "cronologica": "treino até a gravação da crise floor(n/2); teste nas gravações posteriores",
        "modelo": "regressão logística, C=1, class_weight=balanced",
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
