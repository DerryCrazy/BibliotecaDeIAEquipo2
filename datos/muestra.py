"""Muestra: representa un dato individual (atributos + etiqueta)."""
import numpy as np


class Muestra:
    def __init__(self, atributos, etiqueta=None, nombres_atributos=None):
        self._atributos = np.asarray(atributos, dtype=float)
        self._etiqueta = etiqueta
        self._nombres = list(nombres_atributos) if nombres_atributos is not None else None

    @property
    def atributos(self):
        return self._atributos

    @property
    def etiqueta(self):
        return self._etiqueta

    def como_diccionario(self):
        nombres = self._nombres or [f"x{i}" for i in range(len(self._atributos))]
        return dict(zip(nombres, self._atributos))

    def __repr__(self):
        return f"Muestra(n_atributos={len(self._atributos)}, etiqueta={self._etiqueta})"
