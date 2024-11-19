from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import numpy as np
import time

def entrenar_modelo(X_train, y_train, learning_rate, max_iter):
    modelo = LogisticRegression(solver='saga', max_iter=1, warm_start=True)
    iteraciones = []
    precisiones = []
    perdidas = []  # Pérdida (log_loss) en cada iteración

    for i in range(max_iter):
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_train)
        precision = accuracy_score(y_train, y_pred)
        perdida = log_loss(y_train, modelo.predict_proba(X_train))  # Calcular pérdida
        
        iteraciones.append(i + 1)
        precisiones.append(precision)
        perdidas.append(perdida)
        
        time.sleep(0.01)  # Simular tiempo de entrenamiento

    return modelo, iteraciones, precisiones, perdidas

def predecir_nuevos(modelo, scaler, nuevos_datos):
    nuevos_datos_scaled = scaler.transform(nuevos_datos)
    predicciones = modelo.predict(nuevos_datos_scaled)
    return ["defectuoso" if p == 1 else "no defectuoso" for p in predicciones]