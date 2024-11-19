def predecir_nuevos(modelo, scaler, nuevos_datos):
    nuevos_datos_scaled = scaler.transform(nuevos_datos)
    predicciones = modelo.predict(nuevos_datos_scaled)
    return ["defectuoso" if p == 1 else "no defectuoso" for p in predicciones]
