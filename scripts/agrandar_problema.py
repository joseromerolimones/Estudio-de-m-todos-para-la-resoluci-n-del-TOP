import pandas as pd
import numpy as np
import random


def generar_semanas_no_disponibles(
    min_semana=1,
    max_semana=52,
    min_cantidad=10,
    max_cantidad=40
):
    """
    Genera una cadena tipo "1,2,5,6,8,10,..."
    con un número aleatorio de semanas entre min_cantidad y max_cantidad.
    """
    k = random.randint(min_cantidad, max_cantidad)
    # Evitamos que k sea mayor que el total de semanas posibles
    k = min(k, max_semana - min_semana + 1)
    semanas = sorted(random.sample(range(min_semana, max_semana + 1), k))
    return ",".join(str(s) for s in semanas)


def generar_nodos_aleatorios(
    N,
    guardar_csv_nuevos=None,
    guardar_csv_combinado=None,
    ruta_csv_original="circuitos_f1_beneficios_disponible_simulados.csv"
):
    """
    Genera N filas nuevas con la misma estructura de columnas que el CSV original,
    pero con valores completamente aleatorios (no basados en los datos reales).

    Parámetros
    ----------
    N : int
        Número de nodos a generar.
    guardar_csv_nuevos : str o None
        Si no es None, ruta para guardar un CSV solo con los nodos generados.
    guardar_csv_combinado : str o None
        Si no es None, ruta para guardar un CSV combinado (original + nodos).
    ruta_csv_original : str
        Ruta al CSV original para leer solo la cabecera/estructura.
    """

    # Leemos solo para obtener el orden y nombres de las columnas
    df_original = pd.read_csv(ruta_csv_original)
    columnas = df_original.columns.tolist()

    nuevas_filas = []

    for i in range(1, N + 1):
        fila = {}

        # 1) Campos de texto / identificadores
        fila["circuitId"] = f"nodo_{i}"
        fila["nombre"] = f"Nodo {i}"
        fila["ciudad"] = f"Ciudad_{i}"
        fila["pais"] = f"Pais_{random.randint(1, 50)}"  # nombre genérico

        # 2) Coordenadas aleatorias (latitud y longitud)
        #    Usamos rangos amplios, pero fijos, no basados en el CSV.
        #    Latitud: -90 a 90, Longitud: -180 a 180
        fila["latitud"] = round(np.random.uniform(-90, 90), 6)
        fila["longitud"] = round(np.random.uniform(-180, 180), 6)

        # 3) Campos categóricos de ejemplo (opcionales, puedes cambiar listas)
        mercados_posibles = ["bajo", "medio", "alto", "premium"]
        tipos_posibles = ["Permanente", "Urbano"]

        fila["mercado"] = random.choice(mercados_posibles)
        fila["tipo"] = random.choice(tipos_posibles)

        # 4) Valores económicos aleatorios
        #    Supongamos que cada uno va, por ejemplo, de 0 a 60 millones.
        merchandising = round(np.random.uniform(0, 60), 2)
        imagen = round(np.random.uniform(0, 60), 2)
        patrocinio = round(np.random.uniform(0, 60), 2)
        hospitalidad = round(np.random.uniform(0, 60), 2)
        ticketing = round(np.random.uniform(0, 60), 2)

        fila["merchandising_millones_usd"] = merchandising
        fila["beneficio_imagen_millones_usd"] = imagen
        fila["patrocinio_local_millones_usd"] = patrocinio
        fila["hospitalidad_millones_usd"] = hospitalidad
        fila["ticketing_millones_usd"] = ticketing

        # 5) Beneficio total como suma de los anteriores
        fila["beneficio_total_millones_usd"] = round(
            merchandising + imagen + patrocinio + hospitalidad + ticketing,
            2
        )

        # 6) Semanas no disponibles (cadena "a,b,c,...")
        fila["semanas_no_disponibles"] = generar_semanas_no_disponibles()

        nuevas_filas.append(fila)

    # DataFrame solo con los nuevos nodos (mismas columnas y orden)
    df_nuevos = pd.DataFrame(nuevas_filas, columns=columnas)

    # Guardar CSV solo de nodos nuevos, si se pide
    if guardar_csv_nuevos is not None:
        df_nuevos.to_csv(guardar_csv_nuevos, index=False)

    # Guardar CSV combinado, si se pide
    if guardar_csv_combinado is not None:
        df_combinado = pd.concat([df_original, df_nuevos], ignore_index=True)
        df_combinado.to_csv(guardar_csv_combinado, index=False)

    return df_nuevos


if __name__ == "__main__":
    # Ejemplo de uso:
    # Generar 10 nodos, guardar solo los nuevos y también versión combinada
    df_generados = generar_nodos_aleatorios(
        N=10000,
        guardar_csv_nuevos="nodos_aleatorios.csv",
        guardar_csv_combinado="circuitos_mas_nodos_aleatorios.csv",
        ruta_csv_original="circuitos_f1_beneficios_disponible_simulados.csv"
    )
    print(df_generados.head())
