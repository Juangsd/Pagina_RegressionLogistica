import streamlit as st
import io
import pandas as pd

def explorar_datos(datos):
    # Estilo y encabezado general
    st.markdown("## 🔍 Exploración de Datos")
    st.markdown("---")
    
    # Información general del dataset
    st.markdown("### 📄 Información General")
    buffer = io.StringIO()
    datos.info(buf=buffer)
    info = buffer.getvalue()
    
    # Convertimos la salida de info() en un formato tabular más amigable
    info_lines = info.split("\n")
    columnas = ["Columna", "No Nulos", "Tipo de Dato"]
    resumen_info = []
    for line in info_lines[5:-2]:  # Saltar encabezados y líneas finales innecesarias
        parts = line.split()
        if len(parts) >= 5:
            resumen_info.append([parts[1], parts[3], parts[4]])
    
    if resumen_info:
        info_df = pd.DataFrame(resumen_info, columns=columnas)
        st.dataframe(info_df, use_container_width=True)
    else:
        st.text(info)

    # Mostrar primeras filas del dataset
    st.markdown("### 🧾 Primeras Filas del Dataset")
    st.dataframe(datos.head(), use_container_width=True)
    
    # Resumen de valores faltantes
    st.markdown("### ❗ Valores Faltantes")
    valores_faltantes = datos.isnull().sum()
    if valores_faltantes.sum() > 0:
        st.table(valores_faltantes[valores_faltantes > 0])
    else:
        st.success("No hay valores faltantes en el dataset. 🎉")

    # Descripción estadística
    st.markdown("### 📊 Estadísticas Descriptivas")
    st.dataframe(datos.describe(), use_container_width=True)

    # Gráficos exploratorios básicos
    st.markdown("### 📈 Análisis Visual")
    with st.spinner("Generando gráficos..."):
        # Gráficos básicos si hay columnas numéricas
        columnas_numericas = datos.select_dtypes(include=["number"]).columns
        if len(columnas_numericas) > 0:
            st.write("Distribución de las variables numéricas:")
            st.bar_chart(datos[columnas_numericas].sum())
        else:
            st.warning("No se detectaron columnas numéricas para análisis visual.")

