import streamlit as st
import io

def explorar_datos(datos):
    buffer = io.StringIO()
    datos.info(buf=buffer)
    info = buffer.getvalue()
    
    st.write("Información general del dataset:")
    st.text(info)
    st.write("Primeras filas del dataset:")
    st.write(datos.head())
    st.write("Valores faltantes:")
    st.write(datos.isnull().sum())
    st.write("Descripción estadística de las variables numéricas:")
    st.write(datos.describe())
