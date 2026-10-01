"""Variante 1: Perceptron simple (envuelve sklearn.Perceptron por composicion)."""
from sklearn.linear_model import Perceptron
from sklearn.calibration import CalibratedClassifierCV
from .clasificador_base import ClasificadorBase


class PerceptronSimple(ClasificadorBase):
    def __init__(self, max_iter=1000, semilla=42):
        super().__init__("Perceptron simple")
        # El Perceptron no da probabilidades: se calibran con sigmoide para poder hacer soft voting.
        self._modelo = CalibratedClassifierCV(
            Perceptron(max_iter=max_iter, random_state=semilla), method="sigmoid", cv=5)

    def _ajustar(self, X, y):
        self._modelo.fit(X, y)

    def predecir(self, X):
        self._verificar()
        return self._modelo.predict(X)

    def predecir_proba(self, X):
        self._verificar()
        return self._modelo.predict_proba(X)
