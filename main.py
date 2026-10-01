"""Demostración de la Biblioteca de IA - Equipo 2 (Redes neuronales)."""
import sys
from pathlib import Path

# Añade la carpeta superior (Downloads) al camino de Python
sys.path.append(str(Path(__file__).resolve().parent.parent))

from BibliotecaIA.datos.dataset import Dataset
from BibliotecaIA.modelos.perceptron_simple import PerceptronSimple
from BibliotecaIA.modelos.red_neuronal import RedNeuronal
from BibliotecaIA.patrones.strategy import (EstrategiaMLP_2Capas, EstrategiaMLP_3CapasTanh,
                                            EstrategiaMLP_LogisticaSGD)
from BibliotecaIA.ensemble.combinador import CombinadorSoftVoting
from BibliotecaIA.resultados.resultado import Resultado
from BibliotecaIA.resultados.evaluador import Evaluador


def evaluar(modelo, X, y):
    return Resultado(modelo.nombre, y, modelo.predecir(X), modelo.predecir_proba(X)[:, 1],
                     modelo.tiempo_entrenamiento)


def main():
    ds = Dataset().cargar()
    print(ds.resumen())
    Xtr, Xte, ytr, yte = ds.dividir()

    perceptron = PerceptronSimple()
    mlps = [RedNeuronal(EstrategiaMLP_2Capas()),
            RedNeuronal(EstrategiaMLP_3CapasTanh()),
            RedNeuronal(EstrategiaMLP_LogisticaSGD())]
    variantes = [perceptron] + mlps

    # Polimorfismo: mismo metodo entrenar(), comportamiento distinto en cada variante
    for v in variantes:
        v.entrenar(Xtr, ytr)
        print(f"{v.nombre:40s} entrenado en {v.tiempo_entrenamiento:.3f}s")

    soft_mlp = CombinadorSoftVoting(mlps, nombre="Soft Voting (3 MLP)").marcar_entrenado()
    soft_todos = CombinadorSoftVoting(variantes, nombre="Soft Voting (Perceptron + 3 MLP)").marcar_entrenado()

    ev = Evaluador()
    resultados = [evaluar(m, Xte, yte) for m in variantes + [soft_mlp, soft_todos]]
    print(ev.tabla(resultados).to_string())
    print("Matriz de confusion (Soft Voting 3 MLP):\n", ev.matriz_confusion(resultados[-2]))


if __name__ == "__main__":
    main()
