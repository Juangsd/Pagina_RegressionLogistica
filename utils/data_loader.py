import pandas as pd

def cargar_datos(ruta_archivo):
    datos = pd.read_excel(ruta_archivo)
    return datos
