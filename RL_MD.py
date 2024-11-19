import streamlit as st
import numpy as np
import pandas as pd
import datetime  # Para registrar la fecha y hora
from utils.data_loader import cargar_datos
from utils.exploration import explorar_datos
from utils.preprocessor import preprocesar_datos
from utils.model import entrenar_modelo, predecir_nuevos
import matplotlib.pyplot as plt

# Función para registrar predicciones en un archivo CSV
def registrar_prediccion(datos_entrada, prediccion):
    # Crear un DataFrame con los datos de la predicción
    registro = {
        "Fecha y Hora": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Cantidad de Productos": [datos_entrada[0][0]],
        "Tiempo de Entrega": [datos_entrada[0][1]],
        "Predicción": [prediccion[0]]
    }
    df_registro = pd.DataFrame(registro)

    # Guardar en un archivo CSV
    archivo_registro = "registro_predicciones.csv"
    try:
        # Si el archivo ya existe, añadir el nuevo registro
        df_registro.to_csv(archivo_registro, mode='a', header=not pd.io.common.file_exists(archivo_registro), index=False)
    except Exception as e:
        st.error(f"Error al guardar el registro: {e}")

# Función principal
def main():
    # Barra lateral con íconos y navegación
    st.sidebar.title("Navegación 🧭")
    opcion = st.sidebar.radio(
        "Selecciona una opción:",
        [
            "📂 Cargar Datos",
            "🔍 Explorar Datos",
            "⚙️ Entrenar Modelo",
            "📊 Predicción"
        ]
    )

    # Inicializar variables de estado si no existen
    if "modelo" not in st.session_state:
        st.session_state.modelo = None
    if "scaler" not in st.session_state:
        st.session_state.scaler = None
    if "datos" not in st.session_state:
        st.session_state.datos = None

    # Parámetros de la barra lateral
    st.sidebar.header("Parámetros del Modelo")
    tasa_aprendizaje = st.sidebar.slider("Tasa de aprendizaje", 0.01, 1.0, 0.1, 0.01)
    max_iter = st.sidebar.slider("Número máximo de iteraciones", 100, 5000, 1000, 100)

    # Opciones de navegación
    if "📂 Cargar Datos" in opcion:
        st.title("📂 Cargar Datos")
        ruta_archivo = st.file_uploader("Carga un archivo Excel", type=["xlsx"])
        if ruta_archivo:
            datos = cargar_datos(ruta_archivo)
            st.session_state.datos = datos
            st.success("✅ Datos cargados exitosamente")

    elif "🔍 Explorar Datos" in opcion:
        st.title("🔍 Explorar Datos")
        if st.session_state.datos is not None:
            explorar_datos(st.session_state.datos)
        else:
            st.warning("⚠️ Primero debes cargar los datos en la sección 'Cargar Datos'")

    elif "⚙️ Entrenar Modelo" in opcion:
        st.title("⚙️ Entrenar Modelo")
        if st.session_state.datos is not None:
            X_train, X_test, y_train, y_test, scaler = preprocesar_datos(st.session_state.datos)
            if st.button("🚀 Entrenar Modelo"):
                modelo, iteraciones, precisiones, perdidas = entrenar_modelo(X_train, y_train, tasa_aprendizaje, max_iter)
                st.session_state.modelo = modelo
                st.session_state.scaler = scaler

                st.success("✅ Modelo entrenado exitosamente")

                # Gráfico 1: Precisión durante el entrenamiento
                st.subheader("Precisión durante el Entrenamiento")
                fig1, ax1 = plt.subplots()
                ax1.plot(iteraciones, precisiones, marker='o', color='green')
                ax1.set_title("Precisión vs Iteraciones")
                ax1.set_xlabel("Iteraciones")
                ax1.set_ylabel("Precisión")
                st.pyplot(fig1)

                # Gráfico 2: Evolución de la Pérdida
                st.subheader("Evolución de la Pérdida")
                fig2, ax2 = plt.subplots()
                ax2.plot(iteraciones, perdidas, marker='o', color='red')
                ax2.set_title("Pérdida vs Iteraciones")
                ax2.set_xlabel("Iteraciones")
                ax2.set_ylabel("Pérdida (Log-Loss)")
                st.pyplot(fig2)

                # Gráfico 3: Distribución de predicciones
                st.subheader("Distribución de Predicciones")
                y_pred_train = modelo.predict(X_train)
                unique, counts = np.unique(y_pred_train, return_counts=True)
                fig3, ax3 = plt.subplots()
                ax3.bar(unique, counts, color=['blue', 'orange'])
                ax3.set_title("Distribución de Predicciones en el Conjunto de Entrenamiento")
                ax3.set_xticks(unique)
                ax3.set_xticklabels(["No Defectuoso", "Defectuoso"])
                st.pyplot(fig3)
        else:
            st.warning("⚠️ Primero debes cargar los datos en la sección 'Cargar Datos'")

    elif "📊 Predicción" in opcion:
        st.title("📊 Predicción")
        if st.session_state.modelo:
            cantidad_productos = st.number_input("Cantidad de productos en el lote", min_value=0)
            tiempo_entrega = st.number_input("Tiempo de entrega (en minutos)", min_value=0)
            if st.button("🔮 Predecir"):
                nuevos_datos = [[cantidad_productos, tiempo_entrega]]
                prediccion = predecir_nuevos(st.session_state.modelo, st.session_state.scaler, nuevos_datos)
                
                # Registrar la predicción
                registrar_prediccion(nuevos_datos, prediccion)
                
                st.success(f"💡 Predicción: **{prediccion[0]}**")
                st.info("La predicción ha sido registrada.")
        else:
            st.warning("⚠️ Primero debes entrenar el modelo en la sección 'Entrenar Modelo'")

if __name__ == "__main__":
    main()
