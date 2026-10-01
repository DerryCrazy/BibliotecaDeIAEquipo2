"""Variantes 2-4: RedNeuronal (contexto del patron Strategy) sobre MLPClassifier."""
from .clasificador_base import ClasificadorBase
from ..patrones.strategy import EstrategiaMLP


class RedNeuronal(ClasificadorBase):
    def __init__(self, estrategia: EstrategiaMLP, semilla=42):
        super().__init__(estrategia.nombre)
        self._estrategia = estrategia
        self._semilla = semilla
        self._modelo = estrategia.construir(semilla)

    def cambiar_estrategia(self, estrategia: EstrategiaMLP):
        """Intercambia la estrategia en tiempo de ejecucion (requiere reentrenar)."""
        self._estrategia = estrategia
        self._nombre = estrategia.nombre
        self._modelo = estrategia.construir(self._semilla)
        self._entrenado = False

    def _ajustar(self, X, y):
        self._modelo.fit(X, y)

    def predecir(self, X):
        self._verificar()
        return self._modelo.predict(X)

    def predecir_proba(self, X):
        self._verificar()
        return self._modelo.predict_proba(X)

    @property
    def iteraciones(self):
        return self._modelo.n_iter_
