"""Evaluador: calcula metricas a partir de objetos Resultado."""
import pandas as pd
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score, confusion_matrix)


class Evaluador:
    def metricas(self, r):
        return {"Modelo": r.nombre_modelo,
                "ROC-AUC": roc_auc_score(r.y_real, r.y_proba),   # metrica principal
                "Exactitud": accuracy_score(r.y_real, r.y_pred),
                "Precision": precision_score(r.y_real, r.y_pred),
                "Recall": recall_score(r.y_real, r.y_pred),
                "F1": f1_score(r.y_real, r.y_pred)}

    def matriz_confusion(self, r):
        return confusion_matrix(r.y_real, r.y_pred)

    def tabla(self, resultados):
        df = pd.DataFrame([self.metricas(r) for r in resultados]).set_index("Modelo")
        return df.round(4)
