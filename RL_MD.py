import streamlit as st
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from utils.data_loader import cargar_datos
from utils.exploration import explorar_datos
from utils.preprocessor import preprocesar_datos
from utils.model import entrenar_modelo, predecir_nuevos
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sistema de Predicción Flexible", layout="wide", page_icon="🔮")


# Función para registrar predicciones con interpretaciones
def registrar_prediccion(datos_entrada, prediccion):
    registro = {
        "Fecha y Hora": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        **{f"Variable {i+1}": [dato] for i, dato in enumerate(datos_entrada[0])},
        "Predicción": [prediccion[0]],
        "Interpretación": ["Defectuoso" if prediccion[0] == 1 else "No defectuoso"]
    }
    df_registro = pd.DataFrame(registro)
    archivo_registro = "registro_predicciones.csv"
    try:
        if os.path.exists(archivo_registro):
            df_registro.to_csv(archivo_registro, mode='a', header=False, index=False)
        else:
            df_registro.to_csv(archivo_registro, mode='w', header=True, index=False)
    except Exception as e:
        st.error(f"Error al guardar el registro: {e}")
def mostrar_registros():
    archivo_registro = "registro_predicciones.csv"
    try:
        if os.path.exists(archivo_registro):
            # Leer los registros guardados
            registros = pd.read_csv(archivo_registro)
            st.subheader("📜 Registros de Predicciones")
            st.dataframe(registros, use_container_width=True)

            st.markdown("### Análisis Gráfico de Predicciones")
            col1, col2 = st.columns(2)

            # Gráfico de distribuciones de predicciones
            with col1:
                st.subheader("Distribución de Predicciones")
                fig1, ax1 = plt.subplots()
                sns.countplot(data=registros, x="Interpretación", hue="Interpretación", palette="viridis", ax=ax1, legend=False)
                ax1.set_title("Distribución de Lotes (Defectuoso vs No Defectuoso)")
                st.pyplot(fig1)
                plt.close(fig1)

            # Gráfico de series temporales de predicciones
            with col2:
                st.subheader("Predicciones a lo Largo del Tiempo")
                registros["Fecha y Hora"] = pd.to_datetime(registros["Fecha y Hora"])
                registros.sort_values("Fecha y Hora", inplace=True)
                fig2, ax2 = plt.subplots()
                sns.lineplot(
                    data=registros,
                    x="Fecha y Hora",
                    y="Predicción",
                    marker="o",
                    ax=ax2,
                    label="Predicción"
                )
                ax2.set_title("Evolución de Predicciones en el Tiempo")
                ax2.set_ylabel("Defectuoso (1) / No Defectuoso (0)")
                plt.xticks(rotation=45)
                st.pyplot(fig2)
                plt.close(fig2)

            # Interpretación general
            st.markdown("### Interpretación General de Datos")
            defectuosos = registros[registros["Predicción"] == 1].shape[0]
            no_defectuosos = registros[registros["Predicción"] == 0].shape[0]
            total = defectuosos + no_defectuosos

            st.write(f"- **Total de registros**: {total}")
            st.write(f"- **Lotes defectuosos**: {defectuosos}")
            st.write(f"- **Lotes no defectuosos**: {no_defectuosos}")
            
            if total > 0:
                porcentaje_defectuosos = (defectuosos / total) * 100
                st.write(f"- **Porcentaje de lotes defectuosos**: {porcentaje_defectuosos:.2f}%")
                if porcentaje_defectuosos > 50:
                    st.warning("⚠️ **Alerta**: Más de la mitad de los lotes analizados son defectuosos. Considera revisar el proceso de producción.")
                else:
                    st.success("✅ **Buen indicador**: La mayoría de los lotes no son defectuosos.")

        else:
            st.info("No hay registros guardados para analizar.")
    except Exception as e:
        st.error(f"Error al cargar los registros: {e}") 

# Función principal
def main():
    st.sidebar.title("Navegación Principal")
    opcion = st.sidebar.selectbox(
        "Selecciona una opción:",
        [
            "Inicio",
            "Cargar Datos",
            "Explorar Datos",
            "Entrenar Modelo",
            "Predicción",
            "Registros"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.write("💡 **Tips**: Usa las pestañas para explorar funcionalidades.")

    # Variables de estado
    if "modelo" not in st.session_state:
        st.session_state.modelo = None
    if "scaler" not in st.session_state:
        st.session_state.scaler = None
    if "datos" not in st.session_state:
        st.session_state.datos = None

    if opcion == "Inicio":
        st.title("🔮 Sistema de Predicción Flexible")
        st.write("Bienvenido al sistema interactivo para el análisis y predicción basado en datos personalizados.")
        st.image("https://source.unsplash.com/800x400/?data,ai", caption="Optimiza tus decisiones con tecnología avanzada.")
        st.markdown("---")
        st.info("Navega por las secciones en el menú lateral para cargar datos, explorar, entrenar un modelo o realizar predicciones.")

    elif opcion == "Cargar Datos":
        st.title("📂 Cargar Datos")
        st.markdown("Sube tu archivo de datos en formato **Excel, CSV o JSON**:")
        ruta_archivo = st.file_uploader("", type=["xlsx", "csv", "json"])
        if ruta_archivo:
            try:
                datos = cargar_datos(ruta_archivo)
                st.session_state.datos = datos
                st.success("✅ Datos cargados exitosamente")
                st.dataframe(datos.head(), use_container_width=True)
            except Exception as e:
                st.error(f"⚠️ No se pudo cargar el archivo: {e}")

    elif opcion == "Explorar Datos":
        st.title("🔍 Explorar Datos")
        if st.session_state.datos is not None:
            explorar_datos(st.session_state.datos)
        else:
            st.warning("⚠️ Primero debes cargar los datos en la sección 'Cargar Datos'.")

    elif opcion == "Entrenar Modelo":
        st.title("⚙️ Entrenar Modelo")
        if st.session_state.datos is not None:
            X_train, X_test, y_train, y_test, scaler = preprocesar_datos(st.session_state.datos)

            with st.expander("Ajustar Parámetros del Modelo"):
                tasa_aprendizaje = st.slider("Tasa de Aprendizaje", 0.001, 1.0, 0.01, 0.001)
                max_iter = st.slider("Número Máximo de Iteraciones", 100, 10000, 1000, 100)

            if st.button("🚀 Entrenar Modelo"):
                with st.spinner("Entrenando el modelo..."):
                    modelo, iteraciones, precisiones, perdidas = entrenar_modelo(X_train, y_train, tasa_aprendizaje, max_iter)
                st.session_state.modelo = modelo
                st.session_state.scaler = scaler
                st.success("✅ Modelo entrenado exitosamente")

                st.markdown("### Gráficos de Entrenamiento")
                col1, col2 = st.columns(2)

                # Gráfico de Precisión
                with col1:
                    st.subheader("Precisión durante el Entrenamiento")
                    fig1, ax1 = plt.subplots()
                    ax1.plot(iteraciones, precisiones, marker='o', color='green')
                    ax1.set_xlabel("Iteraciones")
                    ax1.set_ylabel("Precisión")
                    st.pyplot(fig1)
                    plt.close(fig1)

                # Gráfico de Pérdidas
                with col2:
                    st.subheader("Evolución de la Pérdida")
                    fig2, ax2 = plt.subplots()
                    ax2.plot(iteraciones, perdidas, marker='o', color='red')
                    ax2.set_xlabel("Iteraciones")
                    ax2.set_ylabel("Pérdida")
                    st.pyplot(fig2)
                    plt.close(fig2)

        else:
            st.warning("⚠️ Primero debes cargar los datos en la sección 'Cargar Datos'.")

    elif opcion == "Predicción":
        st.title("📊 Predicción")
        if st.session_state.modelo:
            st.markdown("Introduce los datos para realizar una predicción:")
            cantidad_productos = st.number_input("Cantidad de productos en el lote:", min_value=0)
            tiempo_entrega = st.number_input("Tiempo de entrega (en minutos):", min_value=0)

            if st.button("🔮 Predecir"):
                nuevos_datos = [[cantidad_productos, tiempo_entrega]]
                prediccion = predecir_nuevos(st.session_state.modelo, st.session_state.scaler, nuevos_datos)
                registrar_prediccion(nuevos_datos, prediccion)
                st.success(f"💡 Predicción: **{prediccion[0]}**")
                st.write("💡 **Interpretación**: El modelo ha determinado si el lote es defectuoso o no, en base a los datos ingresados.")
        else:
            st.warning("⚠️ Primero debes entrenar el modelo en la sección 'Entrenar Modelo'.")

    elif opcion == "Registros":
        st.title("📜 Registros de Predicciones")
        mostrar_registros()


if __name__ == "__main__":
    main()