import requests
import random
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Jolpica API
URL = "https://api.jolpi.ca/ergast/f1/circuits.json?limit=1000"
OUTPUT_CSV = "circuitos_f1_beneficios_disponible_simulados.csv"


def crear_sesion() -> requests.Session:
    session = requests.Session()

    retries = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python Requests",
            "Accept": "application/json",
        }
    )

    return session


def obtener_circuitos() -> list:
    session = crear_sesion()

    try:
        response = session.get(URL, timeout=(10, 30))
        response.raise_for_status()
        data = response.json()

        circuits = data.get("MRData", {}).get("CircuitTable", {}).get("Circuits", [])

        if not circuits:
            raise ValueError("La API respondió, pero no devolvió circuitos.")

        return circuits

    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Timeout al conectar con la API de circuitos. "
            "La API tardó demasiado en responder."
        )
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Error de conexión con la API: {e}")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Error HTTP al consultar la API: {e}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error general al consultar la API: {e}")
    except ValueError as e:
        raise RuntimeError(f"Respuesta inválida de la API: {e}")


def clasificar_mercado(pais: str) -> str:
    mercados_premium = [
        "United Arab Emirates",
        "Qatar",
        "Monaco",
        "Singapore",
        "Saudi Arabia",
        "USA",
        "Japan",
        "Australia",
    ]
    mercados_altos = [
        "UK",
        "Great Britain",
        "Italy",
        "Spain",
        "Canada",
        "Austria",
        "Belgium",
        "Netherlands",
        "Germany",
        "France",
    ]

    if pais in mercados_premium:
        return "premium"
    elif pais in mercados_altos:
        return "alto"
    else:
        return "medio"


def es_urbano(nombre: str) -> bool:
    urbanos = ["street", "city", "marina", "park"]
    nombre = nombre.lower()
    return any(termino in nombre for termino in urbanos)


def generar_semanas_no_disponibles() -> str:
    """
    Las semanas pueden estar entre 1 y 52.
    Retorna un string con formato: 4,6,33,50,51,52
    """
    num_semanas = random.randint(25, 50)
    semanas = set()

    while len(semanas) < num_semanas:
        semana = random.randint(1, 52)
        semanas.add(semana)

    semanas_ordenadas = sorted(semanas)
    return ",".join(map(str, semanas_ordenadas))


def generar_datos_circuitos(circuits: list) -> pd.DataFrame:
    circuitos_datos = []

    for circuit in circuits:
        location = circuit.get("Location", {})

        pais = location.get("country", "Unknown")
        nombre = circuit.get("circuitName", "N/A")
        ciudad = location.get("locality", "N/A")
        latitud = location.get("lat", None)
        longitud = location.get("long", None)

        mercado = clasificar_mercado(pais)
        urbano = es_urbano(nombre)

        if mercado == "premium":
            base_merchandising = random.uniform(8, 15)
            base_imagen = random.uniform(25, 45)
            base_patrocinio_local = random.uniform(15, 35)
            base_hospitalidad = random.uniform(20, 40)
            base_ticketing = random.uniform(30, 60)
        elif mercado == "alto":
            base_merchandising = random.uniform(4, 8)
            base_imagen = random.uniform(12, 25)
            base_patrocinio_local = random.uniform(8, 18)
            base_hospitalidad = random.uniform(10, 22)
            base_ticketing = random.uniform(15, 35)
        else:
            base_merchandising = random.uniform(2, 5)
            base_imagen = random.uniform(5, 12)
            base_patrocinio_local = random.uniform(3, 10)
            base_hospitalidad = random.uniform(5, 12)
            base_ticketing = random.uniform(8, 20)

        if urbano:
            factor_urbano = random.uniform(1.1, 1.3)
            base_merchandising *= factor_urbano
            base_imagen *= factor_urbano
            base_patrocinio_local *= factor_urbano

        beneficio_total = (
            base_merchandising
            + base_imagen
            + base_patrocinio_local
            + base_hospitalidad
            + base_ticketing
        )

        circuito_info = {
            "circuitId": circuit.get("circuitId", "N/A"),
            "nombre": nombre,
            "ciudad": ciudad,
            "pais": pais,
            "latitud": latitud,
            "longitud": longitud,
            "mercado": mercado,
            "tipo": "Urbano" if urbano else "Permanente",
            "merchandising_millones_usd": round(base_merchandising, 2),
            "beneficio_imagen_millones_usd": round(base_imagen, 2),
            "patrocinio_local_millones_usd": round(base_patrocinio_local, 2),
            "hospitalidad_millones_usd": round(base_hospitalidad, 2),
            "ticketing_millones_usd": round(base_ticketing, 2),
            "beneficio_total_millones_usd": round(beneficio_total, 2),
            "semanas_no_disponibles": generar_semanas_no_disponibles(),
        }

        circuitos_datos.append(circuito_info)

    return pd.DataFrame(circuitos_datos)


def main() -> None:
    try:
        print("Obteniendo circuitos desde la API...")
        circuits = obtener_circuitos()
        print(f"Circuitos obtenidos: {len(circuits)}")

        df = generar_datos_circuitos(circuits)
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

        print("\nDatos guardados en:")
        print(OUTPUT_CSV)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()