"""Patron STRATEGY: cada configuracion de MLP es una estrategia intercambiable.

Problema que resuelve: la red (contexto) necesita entrenarse con distintas arquitecturas /
funciones de activacion / optimizadores. Sin Strategy habria una clase por configuracion
o una cadena de if/else en RedNeuronal; con Strategy se agregan configuraciones nuevas
sin modificar la clase RedNeuronal (principio abierto/cerrado).
"""
from abc import ABC, abstractmethod
from sklearn.neural_network import MLPClassifier


class EstrategiaMLP(ABC):
    nombre = "Estrategia"

    @abstractmethod
    def construir(self, semilla=42):
        """Devuelve el modelo de scikit-learn configurado segun la estrategia."""


class EstrategiaMLP_2Capas(EstrategiaMLP):
    nombre = "MLP 2 capas (64-32, ReLU, Adam)"

    def construir(self, semilla=42):
        return MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                             solver="adam", max_iter=1000, random_state=semilla)


class EstrategiaMLP_3CapasTanh(EstrategiaMLP):
    nombre = "MLP 3 capas (64-32-16, tanh, Adam)"

    def construir(self, semilla=42):
        return MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation="tanh",
                             solver="adam", alpha=1e-3, max_iter=1000, random_state=semilla)


class EstrategiaMLP_LogisticaSGD(EstrategiaMLP):
    nombre = "MLP 1 capa (32, logistic, SGD)"

    def construir(self, semilla=42):
        return MLPClassifier(hidden_layer_sizes=(32,), activation="logistic",
                             solver="sgd", learning_rate_init=0.05, momentum=0.9,
                             max_iter=2000, random_state=semilla)
