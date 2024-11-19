import streamlit as st
import numpy as np
import pandas as pd
import datetime  # Para registrar la fecha y hora
import seaborn as sns  # Para gráficos de la matriz de correlación
from utils.data_loader import cargar_datos
from utils.exploration import explorar_datos
from utils.preprocessor import preprocesar_datos
from utils.model import entrenar_modelo, predecir_nuevos
import matplotlib.pyplot as plt

tasa_aprendizaje = 0.01  # Define una tasa de aprendizaje adecuada
max_iter = 1000          # Define el número máximo de iteraciones


# Función para registrar predicciones en un archivo CSV
def registrar_prediccion(datos_entrada, prediccion):
    registro = {
        "Fecha y Hora": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Cantidad de Productos": [datos_entrada[0][0]],
        "Tiempo de Entrega": [datos_entrada[0][1]],
        "Predicción": [prediccion[0]]
    }
    df_registro = pd.DataFrame(registro)
    archivo_registro = "registro_predicciones.csv"
    try:
        df_registro.to_csv(archivo_registro, mode='a', header=not pd.io.common.file_exists(archivo_registro), index=False)
    except Exception as e:
        st.error(f"Error al guardar el registro: {e}")

# Función para mostrar los registros guardados
def mostrar_registros():
    archivo_registro = "registro_predicciones.csv"
    try:
        if pd.io.common.file_exists(archivo_registro):
            registros = pd.read_csv(archivo_registro)
            st.subheader("📜 Registros de Predicciones")
            st.dataframe(registros)
        else:
            st.info("No hay registros guardados.")
    except Exception as e:
        st.error(f"Error al cargar los registros: {e}")

# Función principal
def main():
    st.sidebar.title("Navegación 🧭")
    opcion = st.sidebar.radio(
        "Selecciona una opción:",
        [
            "📂 Cargar Datos",
            "🔍 Explorar Datos",
            "⚙️ Entrenar Modelo",
            "📊 Predicción",
            "📜 Ver Registros"
        ]
    )

    if "modelo" not in st.session_state:
        st.session_state.modelo = None
    if "scaler" not in st.session_state:
        st.session_state.scaler = None
    if "datos" not in st.session_state:
        st.session_state.datos = None

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
        
        # Definir parámetros del modelo
            tasa_aprendizaje = st.sidebar.number_input("Tasa de Aprendizaje", value=0.01, min_value=0.001, max_value=1.0, step=0.001)
            max_iter = st.sidebar.number_input("Número Máximo de Iteraciones", value=1000, min_value=100, max_value=10000, step=100)
        
            if st.button("🚀 Entrenar Modelo"):
                modelo, iteraciones, precisiones, perdidas = entrenar_modelo(X_train, y_train, tasa_aprendizaje, max_iter)
                st.session_state.modelo = modelo
                st.session_state.scaler = scaler

                st.success("✅ Modelo entrenado exitosamente")
            # Gráficos y análisis siguen aquí


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

                # Matriz de correlación
                st.subheader("Matriz de Correlación")
                correlacion = pd.DataFrame(X_train).corr()
                fig4, ax4 = plt.subplots(figsize=(8, 6))
                sns.heatmap(correlacion, annot=True, fmt=".2f", cmap="coolwarm", ax=ax4)
                ax4.set_title("Matriz de Correlación")
                st.pyplot(fig4)
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
                registrar_prediccion(nuevos_datos, prediccion)
                st.success(f"💡 Predicción: **{prediccion[0]}**")
                st.info("La predicción ha sido registrada.")
        else:
            st.warning("⚠️ Primero debes entrenar el modelo en la sección 'Entrenar Modelo'")

    elif "📜 Ver Registros" in opcion:
        mostrar_registros()

if __name__ == "__main__":
    main()
