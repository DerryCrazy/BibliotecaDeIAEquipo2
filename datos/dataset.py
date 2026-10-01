"""Dataset: carga y gestiona el conjunto de datos completo."""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .muestra import Muestra


class Dataset:
    """Dataset numerico (Breast Cancer Wisconsin: 569 muestras, 30 atributos continuos)."""

    def __init__(self, nombre="breast_cancer"):
        self.nombre = nombre
        self._muestras = []
        self._nombres_atributos = []
        self._nombres_clases = []
        self._escalador = StandardScaler()
        self._split = None

    def cargar(self):
        datos = load_breast_cancer()
        self._nombres_atributos = list(datos.feature_names)
        self._nombres_clases = list(datos.target_names)
        self._muestras = [Muestra(x, y, self._nombres_atributos)
                          for x, y in zip(datos.data, datos.target)]
        return self

    def __len__(self):
        return len(self._muestras)

    def __getitem__(self, i):
        return self._muestras[i]

    @property
    def nombres_clases(self):
        return self._nombres_clases

    def matriz(self, muestras=None):
        muestras = muestras if muestras is not None else self._muestras
        X = np.array([m.atributos for m in muestras])
        y = np.array([m.etiqueta for m in muestras])
        return X, y

    def dividir(self, prueba=0.25, semilla=42):
        """Divide en entrenamiento/prueba (estratificado) y estandariza (ajuste solo con train)."""
        X, y = self.matriz()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=prueba,
                                              random_state=semilla, stratify=y)
        Xtr = self._escalador.fit_transform(Xtr)
        Xte = self._escalador.transform(Xte)
        self._split = (Xtr, Xte, ytr, yte)
        return self._split

    def muestras_prueba(self):
        _, Xte, _, yte = self._split
        return [Muestra(x, y, self._nombres_atributos) for x, y in zip(Xte, yte)]

    def resumen(self):
        X, y = self.matriz()
        return {"muestras": len(self), "atributos": X.shape[1],
                "clases": self._nombres_clases, "distribucion": np.bincount(y).tolist()}
