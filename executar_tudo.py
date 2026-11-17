"""Executa o projeto inteiro, na ordem. Uso: python executar_tudo.py

Tempo aproximado em uma máquina de 32 núcleos: 50 a 60 minutos (as extrações com o
GUDHI, scripts 03 e 10, são as etapas mais demoradas). As etapas 10-15 (multipaciente)
precisam de todos os EDF; use --so-chb01 para rodar só o estudo de caso (scripts 01-09).
"""
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
ETAPAS = [
    "01_validacao_sintetica.py",
    "02_parametros_takens.py",
    "03_extrair_descritores.py",
    "04_analise_temporal.py",
    "05_classificacao.py",
    "06_previsao.py",
    "07_mapper.py",
    "08_sensibilidade.py",
    "09_animacoes.py",
]
ETAPAS_MULTI = [
    "10_multipaciente_extrair.py",
    "11_multipaciente_analise.py",
    "12_avaliacao_supervisionada.py",
    "13_figuras_didaticas.py",
    "14_multicanal.py",
    "15_ml_por_que_falha.py",
]


def main():
    registro = RAIZ / "resultados" / "registro_execucao.txt"
    registro.parent.mkdir(parents=True, exist_ok=True)
    with registro.open("w", encoding="utf-8") as log:
        etapas = ETAPAS if "--so-chb01" in sys.argv else ETAPAS + ETAPAS_MULTI
        for etapa in etapas:
            t0 = time.time()
            print(f"\n=== {etapa} ===", flush=True)
            r = subprocess.run([sys.executable, str(RAIZ / "scripts" / etapa)], cwd=RAIZ,
                               capture_output=True, text=True, encoding="utf-8", errors="replace",
                               env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
            saida = "\n".join(l for l in r.stdout.splitlines() if not l.startswith("Reading "))
            print(saida)
            log.write(f"\n=== {etapa} ({time.time() - t0:.0f} s) ===\n{saida}\n")
            if r.returncode != 0:
                print(r.stderr)
                log.write(r.stderr)
                sys.exit(f"Falha em {etapa}")
    # versões dos pacotes (boas práticas, Seção 6.2 do enunciado)
    congelado = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    # pacotes instalados de arquivo local viram só o nome (não publica caminhos da máquina)
    linhas = [l.split(" @ ")[0] if "file:///" in l else l for l in congelado.stdout.splitlines()]
    (RAIZ / "resultados" / "pip_freeze.txt").write_text(
        f"# Python {sys.version}\n" + "\n".join(linhas) + "\n", encoding="utf-8")
    print("\nConcluído. Resultados em resultados/")


if __name__ == "__main__":
    main()
