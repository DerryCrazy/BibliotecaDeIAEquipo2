"""Clase base comun (interfaz) de la familia de redes neuronales."""
from abc import ABC, abstractmethod
import time


class ClasificadorBase(ABC):
    """Interfaz compartida: entrenar(), predecir(), predecir_proba()."""

    def __init__(self, nombre):
        self._nombre = nombre
        self._entrenado = False
        self.tiempo_entrenamiento = 0.0

    @property
    def nombre(self):
        return self._nombre

    def entrenar(self, X, y):
        t0 = time.perf_counter()
        self._ajustar(X, y)
        self.tiempo_entrenamiento = time.perf_counter() - t0
        self._entrenado = True
        return self

    def _verificar(self):
        if not self._entrenado:
            raise RuntimeError(f"El modelo '{self._nombre}' no ha sido entrenado.")

    @abstractmethod
    def _ajustar(self, X, y):
        """Cada variante entrena a su manera (polimorfismo)."""

    @abstractmethod
    def predecir(self, X):
        """Devuelve etiquetas de clase."""

    @abstractmethod
    def predecir_proba(self, X):
        """Devuelve probabilidades (n_muestras, n_clases)."""

    def __repr__(self):
        return f"{self.__class__.__name__}('{self._nombre}')"
