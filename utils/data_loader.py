import pandas as pd

def cargar_datos(ruta_archivo):
    import pandas as pd
    extension = ruta_archivo.name.split('.')[-1]
    if extension == "csv":
        return pd.read_csv(ruta_archivo)
    elif extension == "xlsx":
        return pd.read_excel(ruta_archivo)
    elif extension == "json":
        return pd.read_json(ruta_archivo)
    else:
        raise ValueError("Formato de archivo no soportado. Usa CSV, Excel o JSON.")
