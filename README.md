# Biblioteca de IA — Equipo 2 (Redes neuronales)
Proyecto · Programación para IA · TecNM Cd. Valles · Ago–Dic 2026

**Familia:** Perceptrón simple + MLP (3 configuraciones) · **Patrón:** Strategy · **Combinación:** Soft voting · **Dataset:** Breast Cancer Wisconsin (569 × 30 atributos continuos) · **Métrica principal:** ROC-AUC

## Estructura
```
BibliotecaIA/
├── datos/        muestra.py, dataset.py
├── modelos/      clasificador_base.py, perceptron_simple.py, red_neuronal.py
├── patrones/     strategy.py
├── ensemble/     combinador.py
├── resultados/   resultado.py, evaluador.py
├── main.py       demostración por consola
└── demo.ipynb    notebook de demostración
```
Ejecutar: desde la carpeta que contiene `BibliotecaIA/` → `python -m BibliotecaIA.main`

## Clases
| Clase | Rol |
|---|---|
| `Muestra` | Dato individual: atributos + etiqueta (encapsulados con `@property`). |
| `Dataset` | Carga, gestiona, divide (estratificado) y estandariza (ajuste solo con train). |
| `ClasificadorBase` (abstracta) | Interfaz: `entrenar()`, `predecir()`, `predecir_proba()`; mide tiempo de entrenamiento. |
| `PerceptronSimple` | Variante 1. Envuelve `Perceptron` de sklearn (calibrado con sigmoide para dar probabilidades). |
| `RedNeuronal` | Variantes 2–4. **Contexto** de Strategy; envuelve `MLPClassifier`. |
| `EstrategiaMLP` + 3 concretas | `2Capas` (64-32, ReLU, Adam), `3CapasTanh` (64-32-16, tanh, Adam), `LogisticaSGD` (32, logistic, SGD). |
| `CombinadorSoftVoting` | Promedio (ponderable) de probabilidades; hereda de `ClasificadorBase`. |
| `Resultado` | Predicción, valor real y probabilidad de cada muestra evaluada. |
| `Evaluador` | ROC-AUC, exactitud, precisión, recall, F1, matriz de confusión. |

## Patrón Strategy (justificación)
Las configuraciones de la red (arquitectura, activación, optimizador) cambian, pero el flujo de entrenamiento/predicción no. Strategy encapsula cada configuración en su propia clase, de modo que `RedNeuronal(estrategia=...)` se comporta igual con cualquiera y se puede cambiar en ejecución (`cambiar_estrategia`). Agregar una configuración nueva no obliga a modificar `RedNeuronal` (abierto/cerrado) ni a usar cadenas de `if/else`.

## Polimorfismo
`entrenar()` y `predecir_proba()` se invocan igual sobre todas las variantes, pero cada una responde distinto: el perceptrón calibra con validación cruzada, los MLP optimizan por retropropagación con arquitecturas/activaciones distintas, y el combinador entrena a sus miembros y promedia. Ver el ciclo `for v in variantes: v.entrenar(...)` en `main.py`.

## Soft voting
Para cada muestra, p̂(clase) = Σ wᵢ · pᵢ(clase) con Σ wᵢ = 1 (pesos iguales por defecto). La clase final es el `argmax`. Se reportan dos combinaciones: 3 MLP (la pedida) y Perceptrón + 3 MLP.

## Resultados (conjunto de prueba, 143 muestras, semilla 42)
| Modelo | ROC-AUC | Exactitud | Precisión | Recall | F1 |
|---|---|---|---|---|---|
| Perceptrón simple | 0.9971 | 0.979 | 0.9677 | 1.0000 | 0.9836 |
| MLP 2 capas (64-32, ReLU, Adam) | 0.9960 | 0.972 | 0.9886 | 0.9667 | 0.9775 |
| MLP 3 capas (64-32-16, tanh, Adam) | 0.9956 | 0.979 | 0.9780 | 0.9889 | 0.9834 |
| MLP 1 capa (32, logistic, SGD) | 0.9971 | 0.965 | 0.9885 | 0.9556 | 0.9718 |
| **Soft Voting (3 MLP)** | 0.9964 | **0.986** | 0.9889 | 0.9889 | **0.9889** |
| Soft Voting (Perceptrón + 3 MLP) | 0.9971 | 0.986 | 0.9889 | 0.9889 | 0.9889 |

## Conclusiones
- Todos los modelos superan 0.995 de ROC-AUC: el dataset es casi linealmente separable, por eso el perceptrón simple compite con los MLP.
- El soft voting mejora la exactitud y el F1 (de ≈0.965–0.979 a 0.986) al compensar errores distintos de cada configuración, aunque el ROC-AUC casi no cambia por estar ya saturado.
- El perceptrón entrena ~10× más rápido; los MLP cuestan más tiempo por el mismo nivel de desempeño.
- Con solo 143 muestras de prueba, las diferencias son de 1–3 muestras: conviene validación cruzada antes de declarar un "mejor" modelo.
- Usar sklearn mediante composición permitió centrar el trabajo en el diseño OO, el patrón y la reutilización.
