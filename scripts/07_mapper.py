"""Tópico avançado - Kepler Mapper (EXPLORATÓRIO) sobre o conjunto de janelas.

Cada ponto é uma janela de 8 s descrita pelos 15 descritores de H1 (padronizados).
  Mapper A: lente = duas primeiras componentes principais dos descritores
            topológicos (todas as janelas, 7 gravações).
  Mapper B: lente = tempo relativo ao início da crise (janelas a até 30 min de
            uma crise), como sugere o enunciado.
O grafo é uma ferramenta de exploração; nenhum teste é feito sobre ele.
"""
import _caminho  # noqa: F401
import json

import kmapper as km
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from tda_eeg import config as cfg
from tda_eeg import estilo, extracao, topologia

MIN_AMOSTRAS = 5


def eps_heuristico(X, k=MIN_AMOSTRAS, quantil=0.90):
    """eps do DBSCAN = quantil 90 % da distância ao k-ésimo vizinho (fixado a priori)."""
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    return float(np.quantile(d[:, -1], quantil))


def construir(lente, X, n_cubos, sobreposicao, eps):
    mapper = km.KeplerMapper(verbose=0)
    grafo = mapper.map(lente, X, cover=km.Cover(n_cubes=n_cubos, perc_overlap=sobreposicao),
                       clusterer=DBSCAN(eps=eps, min_samples=MIN_AMOSTRAS))
    return mapper, grafo


def para_networkx(grafo):
    G = nx.Graph()
    for no, membros in grafo["nodes"].items():
        G.add_node(no, membros=np.array(membros))
    for a, vizinhos in grafo["links"].items():
        for b in vizinhos:
            G.add_edge(a, b)
    return G


def tabela_nos(G, df):
    linhas = []
    for no, dados_no in G.nodes(data=True):
        sub = df.iloc[dados_no["membros"]]
        linhas.append({
            "no": no, "n_janelas": len(sub),
            **{f"frac_{c}": float((sub.classe == c).mean())
               for c in ("interictal", "pre-ictal", "ictal", "pos-ictal")},
            "pers_max_media": sub.h1_pers_max.mean(),
            "pers_total_media": sub.h1_pers_total.mean(),
            "log_variancia_media": sub.log_variancia.mean(),
            "t_rel_crise_medio_s": sub.t_rel_crise.mean(),
            "gravacoes": ",".join(sorted(sub.gravacao.str[-2:].unique())),
        })
    return pd.DataFrame(linhas)


def desenhar(ax, G, pos, valores, titulo, rotulo_cor, cmap=estilo.SEQ, vmin=0, vmax=None):
    tam = np.array([len(G.nodes[n]["membros"]) for n in G])
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=estilo.EIXO, width=0.7)
    sc = ax.scatter([pos[n][0] for n in G], [pos[n][1] for n in G], c=valores,
                    s=14 + 5.5 * np.sqrt(tam), cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolor=estilo.TINTA_2, linewidths=0.4, zorder=3)
    ax.set_title(titulo)
    ax.grid(False)
    cb = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.01)
    cb.set_label(rotulo_cor)
    cb.outline.set_visible(False)
    return sc


def main():
    cfg.garantir_pastas()
    estilo.aplicar()
    df = extracao.carregar(extracao.rotulo_config())
    df = df[df.classe != "transicao"].reset_index(drop=True)
    X = StandardScaler().fit_transform(df[topologia.COLUNAS_TOPO].to_numpy(float))
    eps = eps_heuristico(X)
    resumo = {"n_janelas": len(df), "descritores": topologia.COLUNAS_TOPO,
              "dbscan": {"eps": eps, "min_samples": MIN_AMOSTRAS}}

    # ------------------------------------------------------------------ Mapper A
    pca = PCA(n_components=2, random_state=cfg.SEMENTE)
    lente = pca.fit_transform(X)
    mapper, grafo = construir(lente, X, n_cubos=12, sobreposicao=0.4, eps=eps)
    G = para_networkx(grafo)
    nos = tabela_nos(G, df)
    nos.round(4).to_csv(cfg.DIR_TAB / "mapper_A_nos.csv", index=False)
    resumo["mapper_A"] = {
        "lente": "PCA 2D dos descritores topologicos",
        "variancia_explicada": pca.explained_variance_ratio_.round(4).tolist(),
        "n_cubos": 12, "sobreposicao": 0.4, "nos": G.number_of_nodes(),
        "arestas": G.number_of_edges(), "componentes": nx.number_connected_components(G),
        "janelas_cobertas": int(len(np.unique(np.concatenate(list(grafo["nodes"].values()))))),
    }
    cores = np.c_[(df.classe == "ictal").to_numpy(float), (df.classe == "pre-ictal").to_numpy(float),
                  df.h1_pers_max.to_numpy(), df.h1_pers_total.to_numpy()]
    mapper.visualize(
        grafo, path_html=str(cfg.DIR_RES / "mapper_A_interativo.html"),
        title="Mapper A - janelas de EEG (descritores de H1), lente PCA 2D",
        color_values=cores,
        color_function_name=["fração ictal", "fração pré-ictal", "persistência máxima",
                             "persistência total"],
        custom_tooltips=(df.gravacao + " t=" + df.t_ini.astype(int).astype(str) + "s "
                         + df.classe).to_numpy())

    pos = nx.kamada_kawai_layout(G) if G.number_of_nodes() < 400 else nx.spring_layout(G, seed=cfg.SEMENTE)
    nos = nos.set_index("no").loc[list(G)]
    frac_estado = np.array([df.iloc[G.nodes[n]["membros"]].gravacao.isin(
        ["chb01_02", "chb01_03", "chb01_04"]).mean() for n in G])
    fig, eixos = plt.subplots(2, 2, figsize=(13, 10.5), constrained_layout=True)
    paineis = [
        (100 * nos.frac_ictal, "Janelas ictais", "% do nó"),
        (100 * nos["frac_pre-ictal"], "Janelas pré-ictais (10 min antes)", "% do nó"),
        (100 * nos["frac_pos-ictal"], "Janelas pós-ictais (10 min depois)", "% do nó"),
        (100 * frac_estado, "Janelas das gravações 02-04 (as horas em torno das crises)", "% do nó"),
    ]
    for ax, (v, titulo, rot) in zip(eixos.ravel(), paineis):
        desenhar(ax, G, pos, v.to_numpy() if hasattr(v, "to_numpy") else v, titulo, rot)
        ax.set_xticks([])
        ax.set_yticks([])
        for lado in ax.spines.values():
            lado.set_visible(False)
    fig.suptitle(f"Mapper A: {len(df)} janelas, lente = PCA 2D dos descritores de H1 "
                 f"({G.number_of_nodes()} nós; tamanho ∝ √nº de janelas)")
    estilo.salvar(fig, cfg.DIR_FIG / "07_mapper_A.png")

    # ------------------------------------------------------------------ Mapper B
    perto = (df.t_rel_crise.abs() <= 1800).to_numpy() & df.gravacao.isin(
        df.loc[df.classe == "ictal", "gravacao"].unique()).to_numpy()
    dfb = df[perto].reset_index(drop=True)
    Xb = X[perto]
    pc1 = lente[perto, 0]
    mapper_b, grafo_b = construir(dfb[["t_rel_crise"]].to_numpy(float), Xb, n_cubos=24,
                                  sobreposicao=0.35, eps=eps)
    Gb = para_networkx(grafo_b)
    nos_b = tabela_nos(Gb, dfb)
    nos_b.round(4).to_csv(cfg.DIR_TAB / "mapper_B_nos.csv", index=False)
    resumo["mapper_B"] = {"lente": "tempo relativo ao inicio da crise (s)", "n_janelas": len(dfb),
                          "n_cubos": 24, "sobreposicao": 0.35, "nos": Gb.number_of_nodes(),
                          "arestas": Gb.number_of_edges(),
                          "componentes": nx.number_connected_components(Gb)}
    mapper_b.visualize(
        grafo_b, path_html=str(cfg.DIR_RES / "mapper_B_interativo.html"),
        title="Mapper B - lente: tempo relativo ao início da crise",
        color_values=np.c_[dfb.t_rel_crise.to_numpy() / 60, dfb.h1_pers_max.to_numpy()],
        color_function_name=["tempo relativo à crise (min)", "persistência máxima"],
        custom_tooltips=(dfb.gravacao + " t=" + dfb.t_ini.astype(int).astype(str) + "s "
                         + dfb.classe).to_numpy())

    # nós posicionados por (tempo médio, PC1 média): ramificações = regimes distintos na mesma época
    pos_b = {n: (dfb.t_rel_crise.to_numpy()[d["membros"]].mean() / 60, pc1[d["membros"]].mean())
             for n, d in Gb.nodes(data=True)}
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    estilo.faixa_crise(ax, 0, 40 / 60, None)
    ax.axvline(0, color=estilo.LARANJA, lw=1.4)
    ax.text(0, 0.97, "  início da crise", transform=ax.get_xaxis_transform(), ha="left", va="top",
            fontsize=8.5, color=estilo.TINTA_2)
    v = np.array([dfb.h1_pers_max.to_numpy()[Gb.nodes[n]["membros"]].mean() for n in Gb])
    desenhar(ax, Gb, pos_b, v, "", "persistência máxima média do nó", vmin=None)
    ax.grid(True)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_xlabel("tempo médio das janelas do nó, relativo ao início da crise (min)")
    ax.set_ylabel("1ª componente principal\n(média do nó)")
    ax.set_title(f"Mapper B: lente = tempo relativo à crise ({len(dfb)} janelas das 2 gravações "
                 f"com crise, {Gb.number_of_nodes()} nós)")
    estilo.salvar(fig, cfg.DIR_FIG / "07_mapper_B.png")

    (cfg.DIR_TAB / "mapper_resumo.json").write_text(json.dumps(resumo, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in resumo.items() if k != "descritores"}, indent=2))
    mais_ictais = nos.sort_values("frac_ictal", ascending=False).head(5)
    print(mais_ictais[["n_janelas", "frac_ictal", "frac_pre-ictal", "frac_pos-ictal",
                       "pers_max_media", "gravacoes"]].round(3).to_string())


if __name__ == "__main__":
    main()
