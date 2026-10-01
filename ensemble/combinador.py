"""Combinador por SOFT VOTING: promedio de probabilidades de las variantes."""
import numpy as np
from ..modelos.clasificador_base import ClasificadorBase


class CombinadorSoftVoting(ClasificadorBase):
    """Tambien es un ClasificadorBase, por lo que se evalua igual que cualquier variante."""

    def __init__(self, modelos, pesos=None, nombre="Soft Voting"):
        super().__init__(nombre)
        self._modelos = list(modelos)
        self._pesos = np.ones(len(self._modelos)) if pesos is None else np.asarray(pesos, float)
        self._pesos = self._pesos / self._pesos.sum()

    def _ajustar(self, X, y):
        for m in self._modelos:
            m.entrenar(X, y)

    def predecir_proba(self, X):
        self._verificar()
        probas = np.stack([m.predecir_proba(X) for m in self._modelos])   # (k, n, clases)
        return np.tensordot(self._pesos, probas, axes=1)                  # promedio ponderado

    def predecir(self, X):
        return np.argmax(self.predecir_proba(X), axis=1)

    def marcar_entrenado(self):
        """Usar si los modelos ya fueron entrenados por separado."""
        self._entrenado = True
        self.tiempo_entrenamiento = sum(m.tiempo_entrenamiento for m in self._modelos)
        return self
