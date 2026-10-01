"""Resultado: almacena prediccion, valor real y probabilidad de cada muestra evaluada."""
import numpy as np


class Resultado:
    def __init__(self, nombre_modelo, y_real, y_pred, y_proba, tiempo_entrenamiento=None):
        self.nombre_modelo = nombre_modelo
        self.y_real = np.asarray(y_real)
        self.y_pred = np.asarray(y_pred)
        self.y_proba = np.asarray(y_proba)          # probabilidad de la clase positiva (1)
        self.tiempo_entrenamiento = tiempo_entrenamiento

    def __len__(self):
        return len(self.y_real)

    def errores(self):
        return np.where(self.y_real != self.y_pred)[0]

    def __repr__(self):
        return f"Resultado('{self.nombre_modelo}', n={len(self)}, errores={len(self.errores())})"
