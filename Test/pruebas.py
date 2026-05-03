from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from PYOMO_FINAL import F1CalendarOptimizer
import AHORRO_FINAL as ahorro

# ============================================================
# CONFIGURACIÓN POR DEFECTO
# ============================================================
DEFAULT_INSTANCES_ROOT = Path("instancias_generadas")
DEFAULT_FULL_DATA_FILE = Path("circuitos_f1_beneficios_disponible_simulados.csv")
OUTPUT_DIR = Path("resultados_experimentos")
TMP_DIR = OUTPUT_DIR / "_tmp_normalized"

DEFAULT_FIRST_CIRCUIT = "avus"      # puede ser circuitId o nombre
DEFAULT_LAST_CIRCUIT = "jarama"     # puede ser circuitId o nombre


# ============================================================
# ARGUMENTOS DE TERMINAL
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta experimentos sobre instancias pequeñas, medias y/o el "
            "fichero completo de datos con los modos pyomo y/o ahorro."
        )
    )

    parser.add_argument(
        "-s", "--small",
        action="store_true",
        help="Ejecutar solo instancias pequeñas",
    )

    parser.add_argument(
        "-m", "--medium",
        action="store_true",
        help="Ejecutar solo instancias medias",
    )

    parser.add_argument(
        "-td", "--total-data",
        action="store_true",
        help="Ejecutar también el archivo completo de datos",
    )

    parser.add_argument(
        "-mode", "--mode",
        nargs="+",
        default=["pyomo", "ahorro"],
        help=(
            "Modo(s) a ejecutar: pyomo, ahorro o ambos. "
            "Ejemplos: -mode pyomo | -mode ahorro | -mode pyomo ahorro | -mode pyomo,ahorro"
        ),
    )

    parser.add_argument(
        "-alfa", "--alfa",
        nargs="+",
        default=["0.5"],
        help=(
            "Valor(es) de alfa para el modo ahorro. "
            "Ejemplos: -alfa 0.5 | -alfa 0.2 0.4 0.7 | -alfa 0.2,0.4,0.7"
        ),
    )

    parser.add_argument(
        "--instances-root",
        type=str,
        default=str(DEFAULT_INSTANCES_ROOT),
        help="Carpeta raíz donde están las instancias pequeñas y medias",
    )

    parser.add_argument(
        "--full-data-file",
        type=str,
        default=str(DEFAULT_FULL_DATA_FILE),
        help="Ruta al CSV completo que se ejecutará cuando uses -td",
    )

    parser.add_argument(
        "--first-circuit",
        type=str,
        default=DEFAULT_FIRST_CIRCUIT,
        help="Circuito fijo inicial (circuitId o nombre). Usa 'none' para no fijar.",
    )

    parser.add_argument(
        "--last-circuit",
        type=str,
        default=DEFAULT_LAST_CIRCUIT,
        help="Circuito fijo final (circuitId o nombre). Usa 'none' para no fijar.",
    )

    return parser.parse_args()


def parse_modes(raw_modes: List[str]) -> List[str]:
    allowed = {"pyomo", "ahorro"}
    parsed: List[str] = []

    for item in raw_modes:
        for token in item.split(","):
            token = token.strip().lower()
            if not token:
                continue
            if token not in allowed:
                raise ValueError(
                    f"Modo no válido: '{token}'. Valores permitidos: pyomo, ahorro"
                )
            if token not in parsed:
                parsed.append(token)

    if not parsed:
        return ["pyomo", "ahorro"]

    return parsed


def parse_alphas(raw_alphas: List[str]) -> List[float]:
    parsed: List[float] = []

    for item in raw_alphas:
        for token in item.split(","):
            token = token.strip()
            if not token:
                continue

            try:
                alpha = float(token)
            except ValueError:
                raise ValueError(
                    f"Valor de alfa no válido: '{token}'. Debe ser numérico."
                )

            if not (0.0 <= alpha <= 1.0):
                raise ValueError(
                    f"Valor de alfa fuera de rango: '{alpha}'. Debe estar en [0, 1]."
                )

            if alpha not in parsed:
                parsed.append(alpha)

    if not parsed:
        return [0.5]

    return parsed


def normalize_optional_arg(value: str | None) -> str | None:
    if value is None:
        return None
    if str(value).strip().lower() in {"none", "null", ""}:
        return None
    return value


def safe_alpha_label(alpha: float | None) -> str:
    if alpha is None:
        return "NA"
    s = f"{alpha:.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def format_alpha(alpha: float | None) -> str:
    if alpha is None:
        return "NA"
    return f"{alpha:.6f}".rstrip("0").rstrip(".")


def build_config_label(mode: str, alpha: float | None) -> str:
    if mode == "ahorro":
        return f"ahorro_a{safe_alpha_label(alpha)}"
    return "pyomo"


# ============================================================
# NORMALIZACIÓN DE COLUMNAS
# ============================================================
def normalize_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace(".", "")
    )


def standardize_instance_csv(input_csv: Path, output_csv: Path) -> Path:
    df = pd.read_csv(input_csv)

    normalized_to_original = {normalize_name(c): c for c in df.columns}
    rename_map: Dict[str, str] = {}

    candidates = {
        "circuitId": ["circuitid", "id", "circuitoid"],
        "nombre": ["nombre", "circuitname", "name"],
        "ciudad": ["ciudad", "city"],
        "pais": ["pais", "country"],
        "latitud": ["latitud", "lat", "latitude"],
        "longitud": ["longitud", "lon", "lng", "long", "longitude"],
        "mercado": ["mercado", "market"],
        "tipo": ["tipo", "type"],
        "merchandising_millones_usd": [
            "merchandisingmillonesusd",
            "merchandising_millones_usd",
        ],
        "beneficio_imagen_millones_usd": [
            "beneficioimagenmillonesusd",
            "beneficio_imagen_millones_usd",
        ],
        "patrocinio_local_millones_usd": [
            "patrociniolocalmillonesusd",
            "patrocinio_local_millones_usd",
        ],
        "hospitalidad_millones_usd": [
            "hospitalidadmillonesusd",
            "hospitalidad_millones_usd",
        ],
        "ticketing_millones_usd": [
            "ticketingmillonesusd",
            "ticketing_millones_usd",
        ],
        "beneficio_total_millones_usd": [
            "beneficiototalmillonesusd",
            "beneficio_total_millones_usd",
            "beneficiototal",
        ],
        "semanas_no_disponibles": [
            "semanasnodisponibles",
            "semanas_no_disponibles",
            "unavailableweeks",
        ],
    }

    for target, options in candidates.items():
        for opt in options:
            if opt in normalized_to_original:
                rename_map[normalized_to_original[opt]] = target
                break

    df = df.rename(columns=rename_map)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    return output_csv


def validate_columns_for_pyomo(df: pd.DataFrame, input_name: str) -> None:
    required = [
        "circuitId",
        "beneficio_total_millones_usd",
        "semanas_no_disponibles",
        "nombre",
        "latitud",
        "longitud",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[PYOMO] Faltan columnas requeridas {missing} en {input_name}. "
            f"Columnas disponibles: {list(df.columns)}"
        )


def validate_columns_for_ahorro(df: pd.DataFrame, input_name: str) -> None:
    required = [
        "circuitId",
        "nombre",
        "ciudad",
        "pais",
        "latitud",
        "longitud",
        "mercado",
        "tipo",
        "merchandising_millones_usd",
        "beneficio_imagen_millones_usd",
        "patrocinio_local_millones_usd",
        "hospitalidad_millones_usd",
        "ticketing_millones_usd",
        "beneficio_total_millones_usd",
        "semanas_no_disponibles",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[AHORRO] Faltan columnas requeridas {missing} en {input_name}. "
            f"Columnas disponibles: {list(df.columns)}"
        )


# ============================================================
# RESOLUCIÓN DE CIRCUITOS
# ============================================================
def _build_lookup(df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str]]:
    id_to_name = dict(zip(df["circuitId"].astype(str), df["nombre"].astype(str)))
    name_to_id = dict(zip(df["nombre"].astype(str), df["circuitId"].astype(str)))
    return id_to_name, name_to_id


def resolve_circuit_id(df: pd.DataFrame, circuit_ref: str | None) -> str | None:
    circuit_ref = normalize_optional_arg(circuit_ref)
    if circuit_ref is None:
        return None

    id_to_name, name_to_id = _build_lookup(df)
    ref_lower = str(circuit_ref).strip().lower()

    for cid in id_to_name:
        if str(cid).strip().lower() == ref_lower:
            return cid

    for name, cid in name_to_id.items():
        if str(name).strip().lower() == ref_lower:
            return cid

    raise ValueError(f"No se encontró el circuito '{circuit_ref}' en la instancia.")


def resolve_circuit_name(df: pd.DataFrame, circuit_ref: str | None) -> str | None:
    circuit_ref = normalize_optional_arg(circuit_ref)
    if circuit_ref is None:
        return None

    id_to_name, name_to_id = _build_lookup(df)
    ref_lower = str(circuit_ref).strip().lower()

    for cid, name in id_to_name.items():
        if str(cid).strip().lower() == ref_lower:
            return name

    for name in name_to_id:
        if str(name).strip().lower() == ref_lower:
            return name

    raise ValueError(f"No se encontró el circuito '{circuit_ref}' en la instancia.")


# ============================================================
# INSTANCIAS
# ============================================================
def classify_instance(file_path: Path, full_data_file: Path | None = None) -> str:
    try:
        if full_data_file is not None and file_path.resolve() == full_data_file.resolve():
            return "Completa"
    except Exception:
        pass

    name = file_path.stem.lower()
    path_s = str(file_path).lower()

    if name.startswith("instancia-s") or "/s/" in path_s or "\\s\\" in path_s:
        return "Pequeña"
    if name.startswith("instancia-m") or "/m/" in path_s or "\\m\\" in path_s:
        return "Intermedia"
    if any(token in name for token in ["completa", "full", "total", "simulados"]):
        return "Completa"

    return "Desconocida"


def get_instance_files(
    root: Path,
    run_small: bool,
    run_medium: bool,
    run_total_data: bool,
    full_data_file: Path | None,
) -> List[Path]:
    files: List[Path] = []

    if run_small:
        for sub in ["s", "small"]:
            d = root / sub
            if d.exists():
                files.extend(sorted(d.glob("*.csv")))

    if run_medium:
        for sub in ["m", "medium"]:
            d = root / sub
            if d.exists():
                files.extend(sorted(d.glob("*.csv")))

    if run_total_data:
        if full_data_file is None:
            raise ValueError("Debes indicar --full-data-file cuando uses -td.")
        if not full_data_file.exists():
            raise FileNotFoundError(
                f"No se encontró el fichero completo: {full_data_file.resolve()}"
            )
        files.append(full_data_file)

    if not files and not run_total_data:
        files = sorted(root.glob("*.csv"))

    unique = []
    seen = set()
    for f in files:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(f)

    return unique


# ============================================================
# UTILIDADES DE SALIDA
# ============================================================
def save_details(
    calendar_df: pd.DataFrame,
    travel_df: pd.DataFrame,
    output_dir: Path,
    run_name: str,
) -> None:
    calendar_df.to_csv(
        output_dir / f"{run_name}_calendar.csv",
        index=False,
        encoding="utf-8",
    )
    travel_df.to_csv(
        output_dir / f"{run_name}_travel_legs.csv",
        index=False,
        encoding="utf-8",
    )


def build_stats(group_name: str, df_group: pd.DataFrame) -> Dict[str, Any]:
    return {
        "grupo": group_name,
        "n_instancias": len(df_group),

        "beneficio_medio": df_group["totalBenefit"].mean(),
        "beneficio_std": df_group["totalBenefit"].std(ddof=0),
        "beneficio_max": df_group["totalBenefit"].max(),
        "beneficio_min": df_group["totalBenefit"].min(),

        "distancia_media_km": df_group["totalDistanceKm"].mean(),
        "distancia_std_km": df_group["totalDistanceKm"].std(ddof=0),
        "distancia_max_km": df_group["totalDistanceKm"].max(),
        "distancia_min_km": df_group["totalDistanceKm"].min(),

        "tiempo_medio_s": df_group["elapsedSeconds"].mean(),
        "tiempo_std_s": df_group["elapsedSeconds"].std(ddof=0),
        "tiempo_max_s": df_group["elapsedSeconds"].max(),
        "tiempo_min_s": df_group["elapsedSeconds"].min(),

        "n_carreras_medio": df_group["nRaces"].mean(),
        "n_carreras_std": df_group["nRaces"].std(ddof=0),
    }


def select_best_row(df: pd.DataFrame) -> pd.Series:
    ordered = df.sort_values(
        ["totalBenefit", "totalDistanceKm", "elapsedSeconds"],
        ascending=[False, True, True],
    )
    return ordered.iloc[0]


def build_mean_row(
    label: str,
    df_group: pd.DataFrame,
    alpha_value: float | None = None,
) -> Dict[str, Any]:
    return {
        "label": label,
        "alfa": alpha_value,
        "n_instancias": len(df_group),
        "beneficio_medio": df_group["totalBenefit"].mean(),
        "distancia_media_km": df_group["totalDistanceKm"].mean(),
        "tiempo_medio_s": df_group["elapsedSeconds"].mean(),
        "n_carreras_medio": df_group["nRaces"].mean(),
        "beneficio_max": df_group["totalBenefit"].max(),
        "beneficio_min": df_group["totalBenefit"].min(),
        "distancia_max_km": df_group["totalDistanceKm"].max(),
        "distancia_min_km": df_group["totalDistanceKm"].min(),
        "tiempo_max_s": df_group["elapsedSeconds"].max(),
        "tiempo_min_s": df_group["elapsedSeconds"].min(),
    }


def select_best_alpha_by_means(df_ahorro: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    rows = []
    for alpha, sub in df_ahorro.groupby("alfa"):
        rows.append({
            "alfa": alpha,
            "n_instancias": len(sub),
            "beneficio_medio": sub["totalBenefit"].mean(),
            "distancia_media_km": sub["totalDistanceKm"].mean(),
            "tiempo_medio_s": sub["elapsedSeconds"].mean(),
            "n_carreras_medio": sub["nRaces"].mean(),
        })

    alpha_means = pd.DataFrame(rows).sort_values(
        ["beneficio_medio", "distancia_media_km", "tiempo_medio_s"],
        ascending=[False, True, True],
    )
    best_alpha = float(alpha_means.iloc[0]["alfa"])
    return best_alpha, alpha_means


# ============================================================
# RUNNER PYOMO
# ============================================================
def run_pyomo_instance(
    csv_file: Path,
    mode_tmp_dir: Path,
    first_circuit_ref: str | None,
    last_circuit_ref: str | None,
    full_data_file: Path | None,
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    normalized_csv = mode_tmp_dir / csv_file.name
    normalized_csv = standardize_instance_csv(csv_file, normalized_csv)

    df = pd.read_csv(normalized_csv)
    validate_columns_for_pyomo(df, csv_file.name)

    first_id = resolve_circuit_id(df, first_circuit_ref)
    last_id = resolve_circuit_id(df, last_circuit_ref)

    t0 = time.perf_counter()
    optimizer = F1CalendarOptimizer(str(normalized_csv))
    results = optimizer.optimize(firstCircuit=first_id, lastCircuit=last_id)
    elapsed = time.perf_counter() - t0

    if results.get("objectiveValue") is not None:
        calendar_df = optimizer.getResultsDataFrame(results)

        travel_rows = []
        for week1, circuit1, week2, circuit2, distance_km in results.get("travelLegs", []):
            name1 = df.loc[df["circuitId"] == circuit1, "nombre"].iloc[0]
            name2 = df.loc[df["circuitId"] == circuit2, "nombre"].iloc[0]
            travel_rows.append({
                "Semana_origen": week1,
                "CircuitoId_origen": circuit1,
                "Circuito_origen": name1,
                "Semana_destino": week2,
                "CircuitoId_destino": circuit2,
                "Circuito_destino": name2,
                "Distancia_km": distance_km,
            })
        travel_df = pd.DataFrame(travel_rows)
    else:
        calendar_df = pd.DataFrame(columns=["Semana", "CircuitId", "Circuito", "Beneficio_M_USD"])
        travel_df = pd.DataFrame(columns=[
            "Semana_origen", "CircuitoId_origen", "Circuito_origen",
            "Semana_destino", "CircuitoId_destino", "Circuito_destino", "Distancia_km"
        ])

    row: Dict[str, Any] = {
        "mode": "pyomo",
        "alfa": None,
        "config_label": build_config_label("pyomo", None),
        "instancia": csv_file.stem,
        "run_name": f"{csv_file.stem}_pyomo",
        "fichero_original": str(csv_file),
        "fichero_normalizado": str(normalized_csv),
        "tipo_instancia": classify_instance(csv_file, full_data_file),
        "status": str(results.get("status")),
        "objectiveValue": results.get("objectiveValue"),
        "totalBenefit": results.get("totalBenefit"),
        "nRaces": len(results.get("selectedCircuits", [])),
        "totalDistanceKm": results.get("totalDistanceKm"),
        "elapsedSeconds": elapsed,
        "firstCircuit": first_id,
        "lastCircuit": last_id,
    }

    return row, results, calendar_df, travel_df


# ============================================================
# RUNNER AHORRO
# ============================================================
def run_ahorro_instance(
    csv_file: Path,
    mode_tmp_dir: Path,
    first_circuit_ref: str | None,
    last_circuit_ref: str | None,
    alpha: float,
    full_data_file: Path | None,
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    normalized_csv = mode_tmp_dir / csv_file.name
    normalized_csv = standardize_instance_csv(csv_file, normalized_csv)

    df = pd.read_csv(normalized_csv)
    validate_columns_for_ahorro(df, csv_file.name)

    first_id = resolve_circuit_id(df, first_circuit_ref)
    last_id = resolve_circuit_id(df, last_circuit_ref)
    first_name = resolve_circuit_name(df, first_circuit_ref)
    last_name = resolve_circuit_name(df, last_circuit_ref)

    if first_name is None or last_name is None:
        raise ValueError("El modo ahorro requiere circuito inicial y final fijos.")

    ahorro.RUTA_CSV_CIRCUITOS = str(normalized_csv)
    ahorro.NOMBRE_CIRCUITO_ORIGEN = first_name
    ahorro.NOMBRE_CIRCUITO_DESTINO = last_name
    ahorro.ALFA = alpha

    t0 = time.perf_counter()

    nodosBase = ahorro.leerCircuitosCsv(str(normalized_csv))
    if not nodosBase:
        raise RuntimeError("No se pudieron cargar circuitos base para el modo ahorro.")

    gestor = ahorro.GestorDistancias(nodosBase)
    origenCoord = ahorro.obtenerCoordCircuito(nodosBase, first_name)
    destinoCoord = ahorro.obtenerCoordCircuito(nodosBase, last_name)

    nodosClientes = ahorro.expandirNodosPorSemana(nodosBase, numSemanas=ahorro.NUM_SEMANAS)
    nodos = ahorro.agregarNodosEspeciales(nodosClientes, origenCoord, destinoCoord)

    origenId = 0
    destinoId = len(nodos) - 1

    origenGlobalIds = [
        idx for idx, nodo in enumerate(nodos)
        if nodo.circuitoBaseNombre == first_name and nodo.semanaAsignada is not None
    ]
    destinoGlobalIds = [
        idx for idx, nodo in enumerate(nodos)
        if nodo.circuitoBaseNombre == last_name and nodo.semanaAsignada is not None
    ]

    if not origenGlobalIds or not destinoGlobalIds:
        raise RuntimeError("No se encontraron nodos expandidos para origen/destino en modo ahorro.")

    origenNodeId = min(origenGlobalIds, key=lambda nodoId: nodos[nodoId].semanaAsignada)
    destinoNodeId = max(destinoGlobalIds, key=lambda nodoId: nodos[nodoId].semanaAsignada)

    semanaMin = nodos[origenNodeId].semanaAsignada
    semanaMax = nodos[destinoNodeId].semanaAsignada

    if semanaMin is None or semanaMax is None:
        raise RuntimeError("No se pudo determinar semana mínima/máxima de las carreras fijas.")

    tiempo_mat = ahorro.crearMatrizTiemposOptimizada(nodos, gestor, ahorro.VEL)

    seed = ahorro.Ruta(0)
    seed.nodos = [origenId, origenNodeId, destinoNodeId, destinoId]
    seed.nodosVisitados = set(seed.nodos)
    seed.circuitosBaseVisitados = {
        nodos[origenNodeId].circuitoBaseNombre or first_name,
        nodos[destinoNodeId].circuitoBaseNombre or last_name,
    }
    seed.recompensaTotal = (
        nodos[origenNodeId].recompensa + nodos[destinoNodeId].recompensa
    )

    estadoOrigen = ahorro.EstadoCalendario.desdeNodo(
        semanaMin, origenNodeId, nodos[origenNodeId].recompensa
    )
    estadoDestino = ahorro.EstadoCalendario.desdeNodo(
        semanaMax, destinoNodeId, nodos[destinoNodeId].recompensa
    )

    estadoSeed = ahorro.fusionarEstadosCalendario(
        estadoOrigen,
        estadoDestino,
        ahorro.MAX_FINES_SEGUIDOS,
        ahorro.NUM_CARRERAS,
    )
    if estadoSeed is None:
        raise RuntimeError(
            "Inviable: las carreras obligatorias violan restricciones en modo ahorro."
        )

    seed.estadoCal = estadoSeed
    seed.tieneMandatorias = True
    seed.tiempoTotal = sum(
        tiempo_mat[seed.nodos[k]][seed.nodos[k + 1]]
        for k in range(len(seed.nodos) - 1)
    )

    if seed.tiempoTotal > ahorro.T_MAX:
        raise RuntimeError("Inviable: la ruta seed excede T_MAX en modo ahorro.")

    rutas: List[ahorro.Ruta] = [seed]

    for clienteId in range(1, len(nodos) - 1):
        nodo = nodos[clienteId]
        if nodo.circuitoBaseNombre in {first_name, last_name}:
            continue

        ruta = ahorro._rutaSingleton(
            clienteId,
            nodos,
            origenId,
            destinoId,
            tiempo_mat,
            len(rutas),
            ahorro.T_MAX,
            semanaMin,
            semanaMax,
        )
        if ruta is not None:
            rutas.append(ruta)

    savingsList: List[Tuple[float, int, int]] = []
    nClientes = len(nodos) - 2

    for i in range(1, nClientes + 1):
        for j in range(i + 1, nClientes + 1):
            savingsList.append((
                ahorro.calcularSavings(i, j, alpha, nodos, tiempo_mat, origenId, destinoId),
                i,
                j,
            ))
            savingsList.append((
                ahorro.calcularSavings(j, i, alpha, nodos, tiempo_mat, origenId, destinoId),
                j,
                i,
            ))

    savingsList.sort(reverse=True, key=lambda item: item[0])

    nodoARuta: Dict[int, ahorro.Ruta] = {}
    for ruta in rutas:
        for nodoId in ruta.nodosVisitados:
            nodoARuta[nodoId] = ruta

    for _, i, j in savingsList:
        rutaI = nodoARuta.get(i)
        rutaJ = nodoARuta.get(j)

        if not rutaI or not rutaJ or rutaI.id == rutaJ.id:
            continue

        if not (rutaI.tieneMandatorias or rutaJ.tieneMandatorias):
            continue

        if not rutaI.circuitosBaseVisitados.isdisjoint(rutaJ.circuitosBaseVisitados):
            continue

        nuevaRuta = ahorro.fusionClarkeWright(
            rutaI,
            rutaJ,
            ahorro.MAX_FINES_SEGUIDOS,
            nodos,
            tiempo_mat,
            ahorro.T_MAX,
            ahorro.NUM_CARRERAS,
            origenId,
            destinoId,
        )

        if nuevaRuta is None:
            continue

        if nuevaRuta.tiempoTotal > ahorro.T_MAX:
            continue

        rutas.remove(rutaI)
        rutas.remove(rutaJ)
        rutas.append(nuevaRuta)

        for nodoId in rutaI.nodosVisitados:
            nodoARuta.pop(nodoId, None)
        for nodoId in rutaJ.nodosVisitados:
            nodoARuta.pop(nodoId, None)
        for nodoId in nuevaRuta.nodosVisitados:
            nodoARuta[nodoId] = nuevaRuta

    rutasMandatorias = [ruta for ruta in rutas if ruta.tieneMandatorias]
    if not rutasMandatorias:
        raise RuntimeError("No quedó ninguna ruta con las carreras obligatorias en modo ahorro.")

    rutasMandatorias.sort(
        key=lambda ruta: ruta.estadoCal.beneficioReal,
        reverse=True,
    )
    solucionFinal = rutasMandatorias[:ahorro.NUM_VEHICULOS]
    ruta_final = solucionFinal[0]

    calendario = ahorro.crearCalendarioDesdeEstado(ruta_final)

    name_to_id = dict(zip(df["nombre"].astype(str), df["circuitId"].astype(str)))
    name_to_benefit = dict(zip(df["nombre"].astype(str), df["beneficio_total_millones_usd"].astype(float)))

    calendar_rows = []
    travel_rows = []
    total_distance_km = 0.0

    semanas_ordenadas = sorted(calendario.keys())
    for idx, semana in enumerate(semanas_ordenadas):
        nodoId = calendario[semana]
        nodo = nodos[nodoId]
        circuit_name = nodo.circuitoBaseNombre or nodo.nombre
        circuit_id = name_to_id.get(circuit_name, circuit_name)
        benefit = float(name_to_benefit.get(circuit_name, nodo.recompensa))

        calendar_rows.append({
            "Semana": int(semana),
            "CircuitId": circuit_id,
            "Circuito": circuit_name,
            "Beneficio_M_USD": benefit,
        })

        if idx > 0:
            prev_week = semanas_ordenadas[idx - 1]
            prev_nodoId = calendario[prev_week]
            prev_nodo = nodos[prev_nodoId]
            prev_name = prev_nodo.circuitoBaseNombre or prev_nodo.nombre
            prev_id = name_to_id.get(prev_name, prev_name)

            distance_km = float(gestor.obtenerDistancia(prev_name, circuit_name))
            total_distance_km += distance_km

            travel_rows.append({
                "Semana_origen": int(prev_week),
                "CircuitoId_origen": prev_id,
                "Circuito_origen": prev_name,
                "Semana_destino": int(semana),
                "CircuitoId_destino": circuit_id,
                "Circuito_destino": circuit_name,
                "Distancia_km": distance_km,
            })

    calendar_df = pd.DataFrame(calendar_rows)
    travel_df = pd.DataFrame(travel_rows)
    total_benefit_exact = float(calendar_df["Beneficio_M_USD"].sum()) if not calendar_df.empty else 0.0
    elapsed = time.perf_counter() - t0

    results: Dict[str, Any] = {
        "status": "HEURISTIC_OK",
        "objectiveValue": float(ruta_final.estadoCal.beneficioReal),
        "selectedCircuits": calendar_df["CircuitId"].tolist(),
        "calendar": [
            (int(r["Semana"]), str(r["CircuitId"]), float(r["Beneficio_M_USD"]))
            for _, r in calendar_df.iterrows()
        ],
        "travelLegs": [
            (
                int(r["Semana_origen"]),
                str(r["CircuitoId_origen"]),
                int(r["Semana_destino"]),
                str(r["CircuitoId_destino"]),
                float(r["Distancia_km"]),
            )
            for _, r in travel_df.iterrows()
        ],
        "totalBenefit": total_benefit_exact,
        "totalDistanceKm": float(total_distance_km),
    }

    run_name = f"{csv_file.stem}_ahorro_a{safe_alpha_label(alpha)}"

    row: Dict[str, Any] = {
        "mode": "ahorro",
        "alfa": alpha,
        "config_label": build_config_label("ahorro", alpha),
        "instancia": csv_file.stem,
        "run_name": run_name,
        "fichero_original": str(csv_file),
        "fichero_normalizado": str(normalized_csv),
        "tipo_instancia": classify_instance(csv_file, full_data_file),
        "status": str(results.get("status")),
        "objectiveValue": results.get("objectiveValue"),
        "totalBenefit": results.get("totalBenefit"),
        "nRaces": len(results.get("selectedCircuits", [])),
        "totalDistanceKm": results.get("totalDistanceKm"),
        "elapsedSeconds": elapsed,
        "firstCircuit": first_id,
        "lastCircuit": last_id,
    }

    return row, results, calendar_df, travel_df


# ============================================================
# PROCESADO DE UN MODO
# ============================================================
def process_mode(
    mode: str,
    instance_files: List[Path],
    first_circuit_ref: str | None,
    last_circuit_ref: str | None,
    alphas: List[float],
    full_data_file: Path | None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    mode_output_dir = OUTPUT_DIR / mode
    mode_tmp_dir = TMP_DIR / mode

    mode_output_dir.mkdir(parents=True, exist_ok=True)
    mode_tmp_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    successful_records: List[Dict[str, Any]] = []

    print()
    print("=" * 120)
    print(f"EJECUCIÓN DEL MODO: {mode.upper()}")
    print("=" * 120)

    runs: List[tuple[Path, float | None]] = []
    if mode == "pyomo":
        runs = [(csv_file, None) for csv_file in instance_files]
    elif mode == "ahorro":
        for alpha in alphas:
            for csv_file in instance_files:
                runs.append((csv_file, alpha))
    else:
        raise ValueError(f"Modo no soportado: {mode}")

    for i, (csv_file, alpha) in enumerate(runs, start=1):
        alpha_text = f" | alfa={format_alpha(alpha)}" if alpha is not None else ""
        print("-" * 120)
        print(f"[{i}/{len(runs)}] [{mode}] Ejecutando: {csv_file.name}{alpha_text}")

        try:
            if mode == "pyomo":
                row, results, calendar_df, travel_df = run_pyomo_instance(
                    csv_file, mode_tmp_dir, first_circuit_ref, last_circuit_ref, full_data_file
                )
            else:
                row, results, calendar_df, travel_df = run_ahorro_instance(
                    csv_file, mode_tmp_dir, first_circuit_ref, last_circuit_ref,
                    alpha=alpha, full_data_file=full_data_file  # type: ignore[arg-type]
                )

            summary_rows.append(row)

            if results.get("objectiveValue") is not None:
                save_details(calendar_df, travel_df, mode_output_dir, row["run_name"])

                successful_records.append({
                    "row": row,
                    "results": results,
                    "calendar_df": calendar_df,
                    "travel_df": travel_df,
                })

                alpha_msg = f" | alfa={format_alpha(row['alfa'])}" if row["alfa"] is not None else ""
                print(
                    f"OK{alpha_msg} | beneficio={row['totalBenefit']:.2f} M USD | "
                    f"distancia={row['totalDistanceKm']:.2f} km | "
                    f"tiempo={row['elapsedSeconds']:.2f} s | "
                    f"carreras={row['nRaces']}"
                )
            else:
                print(f"SIN SOLUCIÓN | status={row['status']}")

        except Exception as e:
            error_row = {
                "mode": mode,
                "alfa": alpha,
                "config_label": build_config_label(mode, alpha),
                "instancia": csv_file.stem,
                "run_name": f"{csv_file.stem}_{build_config_label(mode, alpha)}",
                "fichero_original": str(csv_file),
                "fichero_normalizado": None,
                "tipo_instancia": classify_instance(csv_file, full_data_file),
                "status": f"ERROR: {e}",
                "objectiveValue": None,
                "totalBenefit": None,
                "nRaces": None,
                "totalDistanceKm": None,
                "elapsedSeconds": None,
                "firstCircuit": first_circuit_ref,
                "lastCircuit": last_circuit_ref,
            }
            summary_rows.append(error_row)
            print(f"ERROR en {csv_file.name}: {e}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        mode_output_dir / f"resumen_ejecuciones_{mode}.csv",
        index=False,
        encoding="utf-8",
    )

    ok_df = summary_df[summary_df["objectiveValue"].notna()].copy()
    ok_df = ok_df.sort_values(["totalBenefit", "totalDistanceKm"], ascending=[False, True])
    ok_df.to_csv(
        mode_output_dir / f"ranking_resultados_{mode}.csv",
        index=False,
        encoding="utf-8",
    )

    if not ok_df.empty:
        list_cols = [
            "mode", "alfa", "config_label", "instancia", "tipo_instancia",
            "totalBenefit", "totalDistanceKm", "elapsedSeconds", "nRaces", "status"
        ]
        list_df = ok_df[list_cols].copy()

        list_df.to_csv(
            mode_output_dir / f"lista_beneficios_distancias_tiempos_{mode}.csv",
            index=False,
            encoding="utf-8",
        )

        stats_rows = [build_stats("Global", ok_df)]
        for group_name, group_df in ok_df.groupby("tipo_instancia"):
            stats_rows.append(build_stats(group_name, group_df))

        if mode == "ahorro":
            for alpha, alpha_df in ok_df.groupby("alfa"):
                stats_rows.append(build_stats(f"alfa={format_alpha(alpha)}", alpha_df))

        stats_df = pd.DataFrame(stats_rows)
        stats_df.to_csv(
            mode_output_dir / f"estadisticas_medias_{mode}.csv",
            index=False,
            encoding="utf-8",
        )

        best_row = select_best_row(ok_df)
        best_record = next(
            rec for rec in successful_records
            if rec["row"]["run_name"] == best_row["run_name"]
        )
        best_calendar_df = best_record["calendar_df"]
        best_calendar_df.to_csv(
            mode_output_dir / f"mejor_calendario_{mode}.csv",
            index=False,
            encoding="utf-8",
        )

        print()
        print("=" * 120)
        print(f"LISTA DE RESULTADOS - {mode.upper()}")
        print("=" * 120)
        for _, r in list_df.iterrows():
            alpha_text = format_alpha(r["alfa"]) if pd.notna(r["alfa"]) else "NA"
            print(
                f"{r['instancia']:<18} | {r['tipo_instancia']:<12} | alfa={alpha_text:<6} | "
                f"beneficio={r['totalBenefit']:.2f} M USD | "
                f"distancia={r['totalDistanceKm']:.2f} km | "
                f"tiempo={r['elapsedSeconds']:.2f} s | "
                f"carreras={int(r['nRaces'])}"
            )

        print()
        print("=" * 120)
        print(f"ESTADÍSTICAS MEDIAS - {mode.upper()}")
        print("=" * 120)
        for _, r in stats_df.iterrows():
            print(
                f"{r['grupo']:<18} | "
                f"n={int(r['n_instancias'])} | "
                f"beneficio medio={r['beneficio_medio']:.2f} M USD | "
                f"distancia media={r['distancia_media_km']:.2f} km | "
                f"tiempo medio={r['tiempo_medio_s']:.2f} s | "
                f"carreras medias={r['n_carreras_medio']:.2f}"
            )

        print()
        print("=" * 120)
        print(f"MEJOR RESULTADO - {mode.upper()}")
        print("=" * 120)
        print(f"Instancia: {best_row['instancia']}")
        print(f"Alfa: {format_alpha(best_row['alfa']) if pd.notna(best_row['alfa']) else 'NA'}")
        print(f"Beneficio total: {best_row['totalBenefit']:.2f} M USD")
        print(f"Distancia total: {best_row['totalDistanceKm']:.2f} km")
        print(f"Tiempo: {best_row['elapsedSeconds']:.2f} s")
        print(f"Número de carreras: {best_row['nRaces']}")
    else:
        print(f"\nNo hubo soluciones válidas para el modo {mode}.")

    return summary_df, ok_df, successful_records


# ============================================================
# SALIDAS GLOBALES
# ============================================================
def save_global_outputs(
    all_summary_dfs: List[pd.DataFrame],
    all_successful_records: List[Dict[str, Any]],
) -> None:
    if not all_summary_dfs:
        return

    global_summary = pd.concat(all_summary_dfs, ignore_index=True)
    global_summary.to_csv(
        OUTPUT_DIR / "resumen_global_modos.csv",
        index=False,
        encoding="utf-8",
    )

    global_ok = global_summary[global_summary["objectiveValue"].notna()].copy()
    if global_ok.empty:
        return

    global_ok = global_ok.sort_values(["totalBenefit", "totalDistanceKm"], ascending=[False, True])
    global_ok.to_csv(
        OUTPUT_DIR / "ranking_global_modos.csv",
        index=False,
        encoding="utf-8",
    )

    list_cols = [
        "mode", "alfa", "config_label", "instancia", "tipo_instancia",
        "totalBenefit", "totalDistanceKm", "elapsedSeconds", "nRaces", "status"
    ]
    list_df = global_ok[list_cols].copy()
    list_df.to_csv(
        OUTPUT_DIR / "lista_global_beneficios_distancias_tiempos.csv",
        index=False,
        encoding="utf-8",
    )

    stats_rows = [build_stats("Global", global_ok)]

    for mode_name, mode_df in global_ok.groupby("mode"):
        stats_rows.append(build_stats(f"Modo={mode_name}", mode_df))

    for tipo_name, tipo_df in global_ok.groupby("tipo_instancia"):
        stats_rows.append(build_stats(f"Tipo={tipo_name}", tipo_df))

    for (mode_name, tipo_name), group_df in global_ok.groupby(["mode", "tipo_instancia"]):
        stats_rows.append(build_stats(f"{mode_name}-{tipo_name}", group_df))

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(
        OUTPUT_DIR / "estadisticas_globales_modos.csv",
        index=False,
        encoding="utf-8",
    )

    best_global = select_best_row(global_ok)
    best_record = next(
        rec for rec in all_successful_records
        if rec["row"]["run_name"] == best_global["run_name"]
    )
    best_record["calendar_df"].to_csv(
        OUTPUT_DIR / "mejor_calendario_global.csv",
        index=False,
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print("RESUMEN GLOBAL MULTIMODO")
    print("=" * 120)
    print(f"Mejor configuración: {best_global['config_label']}")
    print(f"Instancia: {best_global['instancia']}")
    print(f"Alfa: {format_alpha(best_global['alfa']) if pd.notna(best_global['alfa']) else 'NA'}")
    print(f"Beneficio total: {best_global['totalBenefit']:.2f} M USD")
    print(f"Distancia total: {best_global['totalDistanceKm']:.2f} km")
    print(f"Tiempo: {best_global['elapsedSeconds']:.2f} s")

    print()
    print("=" * 120)
    print("FICHEROS GLOBALES GENERADOS")
    print("=" * 120)
    print(OUTPUT_DIR / "resumen_global_modos.csv")
    print(OUTPUT_DIR / "ranking_global_modos.csv")
    print(OUTPUT_DIR / "lista_global_beneficios_distancias_tiempos.csv")
    print(OUTPUT_DIR / "estadisticas_globales_modos.csv")
    print(OUTPUT_DIR / "mejor_calendario_global.csv")


# ============================================================
# TABLAS COMPARATIVAS
# ============================================================
def save_comparison_tables(all_summary_dfs: List[pd.DataFrame]) -> None:
    if not all_summary_dfs:
        return

    global_summary = pd.concat(all_summary_dfs, ignore_index=True)
    global_ok = global_summary[global_summary["objectiveValue"].notna()].copy()

    if global_ok.empty:
        return

    pyomo_df = global_ok[global_ok["mode"] == "pyomo"].copy()
    ahorro_df = global_ok[global_ok["mode"] == "ahorro"].copy()

    # ------------------------------------------------------------
    # 1) TABLAS GENERALES PYOMO VS AHORRO
    # ------------------------------------------------------------
    if not pyomo_df.empty and not ahorro_df.empty:
        best_pyomo = select_best_row(pyomo_df)
        best_ahorro = select_best_row(ahorro_df)

        best_compare_df = pd.DataFrame([
            {
                "mode": "pyomo",
                "alfa": None,
                "instancia": best_pyomo["instancia"],
                "tipo_instancia": best_pyomo["tipo_instancia"],
                "totalBenefit": best_pyomo["totalBenefit"],
                "totalDistanceKm": best_pyomo["totalDistanceKm"],
                "elapsedSeconds": best_pyomo["elapsedSeconds"],
                "nRaces": best_pyomo["nRaces"],
                "status": best_pyomo["status"],
            },
            {
                "mode": "ahorro",
                "alfa": best_ahorro["alfa"],
                "instancia": best_ahorro["instancia"],
                "tipo_instancia": best_ahorro["tipo_instancia"],
                "totalBenefit": best_ahorro["totalBenefit"],
                "totalDistanceKm": best_ahorro["totalDistanceKm"],
                "elapsedSeconds": best_ahorro["elapsedSeconds"],
                "nRaces": best_ahorro["nRaces"],
                "status": best_ahorro["status"],
            },
        ])
        best_compare_df.to_csv(
            OUTPUT_DIR / "tabla_comparativa_mejores_soluciones.csv",
            index=False,
            encoding="utf-8",
        )

        best_alpha_global, ahorro_alpha_means = select_best_alpha_by_means(ahorro_df)
        ahorro_best_mean_df = ahorro_df[ahorro_df["alfa"] == best_alpha_global].copy()

        mean_compare_df = pd.DataFrame([
            build_mean_row("pyomo", pyomo_df, None),
            build_mean_row("ahorro", ahorro_best_mean_df, best_alpha_global),
        ])
        mean_compare_df.to_csv(
            OUTPUT_DIR / "tabla_comparativa_medias.csv",
            index=False,
            encoding="utf-8",
        )

        rows_by_type = []
        common_types = sorted(
            set(pyomo_df["tipo_instancia"].dropna().tolist()) &
            set(ahorro_df["tipo_instancia"].dropna().tolist())
        )

        for tipo in common_types:
            py_tipo = pyomo_df[pyomo_df["tipo_instancia"] == tipo].copy()
            ah_tipo = ahorro_df[ahorro_df["tipo_instancia"] == tipo].copy()

            if py_tipo.empty or ah_tipo.empty:
                continue

            best_alpha_tipo, _ = select_best_alpha_by_means(ah_tipo)
            ah_tipo_best = ah_tipo[ah_tipo["alfa"] == best_alpha_tipo].copy()

            rows_by_type.append({
                "tipo_instancia": tipo,

                "pyomo_n_instancias": len(py_tipo),
                "pyomo_beneficio_medio": py_tipo["totalBenefit"].mean(),
                "pyomo_distancia_media_km": py_tipo["totalDistanceKm"].mean(),
                "pyomo_tiempo_medio_s": py_tipo["elapsedSeconds"].mean(),
                "pyomo_n_carreras_medio": py_tipo["nRaces"].mean(),

                "ahorro_alfa_seleccionado": best_alpha_tipo,
                "ahorro_n_instancias": len(ah_tipo_best),
                "ahorro_beneficio_medio": ah_tipo_best["totalBenefit"].mean(),
                "ahorro_distancia_media_km": ah_tipo_best["totalDistanceKm"].mean(),
                "ahorro_tiempo_medio_s": ah_tipo_best["elapsedSeconds"].mean(),
                "ahorro_n_carreras_medio": ah_tipo_best["nRaces"].mean(),
            })

        mean_by_type_df = pd.DataFrame(rows_by_type)
        mean_by_type_df.to_csv(
            OUTPUT_DIR / "tabla_comparativa_medias_por_tipo.csv",
            index=False,
            encoding="utf-8",
        )

        print()
        print("=" * 120)
        print("TABLA COMPARATIVA DE LAS MEJORES SOLUCIONES")
        print("=" * 120)
        print(
            f"{'Método':<12}{'Alfa':<10}{'Instancia':<20}{'Tipo':<15}"
            f"{'Beneficio':>15}{'Distancia (km)':>18}{'Tiempo (s)':>15}{'Carreras':>12}"
        )
        print("-" * 120)
        for _, r in best_compare_df.iterrows():
            print(
                f"{str(r['mode']):<12}"
                f"{format_alpha(r['alfa']) if pd.notna(r['alfa']) else 'NA':<10}"
                f"{str(r['instancia']):<20}"
                f"{str(r['tipo_instancia']):<15}"
                f"{float(r['totalBenefit']):>15.2f}"
                f"{float(r['totalDistanceKm']):>18.2f}"
                f"{float(r['elapsedSeconds']):>15.2f}"
                f"{int(r['nRaces']):>12}"
            )

        print()
        print("=" * 120)
        print("TABLA COMPARATIVA DE MEDIAS")
        print("=" * 120)
        print(
            f"{'Método':<12}{'Alfa':<10}{'Instancias':>12}"
            f"{'Beneficio medio':>20}{'Distancia media':>20}"
            f"{'Tiempo medio':>18}{'Carreras medias':>18}"
        )
        print("-" * 120)
        for _, r in mean_compare_df.iterrows():
            print(
                f"{str(r['label']):<12}"
                f"{format_alpha(r['alfa']) if pd.notna(r['alfa']) else 'NA':<10}"
                f"{int(r['n_instancias']):>12}"
                f"{float(r['beneficio_medio']):>20.2f}"
                f"{float(r['distancia_media_km']):>20.2f}"
                f"{float(r['tiempo_medio_s']):>18.2f}"
                f"{float(r['n_carreras_medio']):>18.2f}"
            )

        if not mean_by_type_df.empty:
            print()
            print("=" * 120)
            print("TABLA COMPARATIVA DE MEDIAS POR TIPO DE INSTANCIA")
            print("=" * 120)
            for _, r in mean_by_type_df.iterrows():
                print(f"\nTipo de instancia: {r['tipo_instancia']}")
                print(
                    f" PYOMO | n={int(r['pyomo_n_instancias'])} | "
                    f"beneficio medio={float(r['pyomo_beneficio_medio']):.2f} | "
                    f"distancia media={float(r['pyomo_distancia_media_km']):.2f} km | "
                    f"tiempo medio={float(r['pyomo_tiempo_medio_s']):.2f} s | "
                    f"carreras medias={float(r['pyomo_n_carreras_medio']):.2f}"
                )
                print(
                    f" AHORRO | alfa={format_alpha(r['ahorro_alfa_seleccionado'])} | "
                    f"n={int(r['ahorro_n_instancias'])} | "
                    f"beneficio medio={float(r['ahorro_beneficio_medio']):.2f} | "
                    f"distancia media={float(r['ahorro_distancia_media_km']):.2f} km | "
                    f"tiempo medio={float(r['ahorro_tiempo_medio_s']):.2f} s | "
                    f"carreras medias={float(r['ahorro_n_carreras_medio']):.2f}"
                )
    else:
        print()
        print("No se han podido generar las tablas pyomo vs ahorro porque falta al menos uno de los dos métodos con soluciones válidas.")

    # ------------------------------------------------------------
    # 2) TABLAS ESPECÍFICAS DE ALFAS DEL AHORRO
    # ------------------------------------------------------------
    if not ahorro_df.empty:
        ahorro_best_by_alpha_rows = []
        for alpha, sub in ahorro_df.groupby("alfa"):
            best = select_best_row(sub)
            ahorro_best_by_alpha_rows.append({
                "alfa": alpha,
                "instancia": best["instancia"],
                "tipo_instancia": best["tipo_instancia"],
                "totalBenefit": best["totalBenefit"],
                "totalDistanceKm": best["totalDistanceKm"],
                "elapsedSeconds": best["elapsedSeconds"],
                "nRaces": best["nRaces"],
                "status": best["status"],
            })

        ahorro_best_by_alpha_df = pd.DataFrame(ahorro_best_by_alpha_rows).sort_values("alfa")
        ahorro_best_by_alpha_df.to_csv(
            OUTPUT_DIR / "tabla_comparativa_alfas_mejores.csv",
            index=False,
            encoding="utf-8",
        )

        ahorro_mean_by_alpha_rows = []
        for alpha, sub in ahorro_df.groupby("alfa"):
            ahorro_mean_by_alpha_rows.append(build_mean_row(f"alfa={format_alpha(alpha)}", sub, alpha))

        ahorro_mean_by_alpha_df = pd.DataFrame(ahorro_mean_by_alpha_rows).sort_values("alfa")
        ahorro_mean_by_alpha_df.to_csv(
            OUTPUT_DIR / "tabla_comparativa_alfas_medias.csv",
            index=False,
            encoding="utf-8",
        )

        ahorro_mean_by_alpha_type_rows = []
        for (tipo, alpha), sub in ahorro_df.groupby(["tipo_instancia", "alfa"]):
            ahorro_mean_by_alpha_type_rows.append({
                "tipo_instancia": tipo,
                "alfa": alpha,
                "n_instancias": len(sub),
                "beneficio_medio": sub["totalBenefit"].mean(),
                "distancia_media_km": sub["totalDistanceKm"].mean(),
                "tiempo_medio_s": sub["elapsedSeconds"].mean(),
                "n_carreras_medio": sub["nRaces"].mean(),
            })

        ahorro_mean_by_alpha_type_df = pd.DataFrame(ahorro_mean_by_alpha_type_rows).sort_values(
            ["tipo_instancia", "alfa"]
        )
        ahorro_mean_by_alpha_type_df.to_csv(
            OUTPUT_DIR / "tabla_comparativa_alfas_medias_por_tipo.csv",
            index=False,
            encoding="utf-8",
        )

        print()
        print("=" * 120)
        print("TABLA COMPARATIVA DE ALFAS - MEJORES SOLUCIONES")
        print("=" * 120)
        for _, r in ahorro_best_by_alpha_df.iterrows():
            print(
                f"alfa={format_alpha(r['alfa']):<6} | "
                f"instancia={r['instancia']:<18} | "
                f"tipo={r['tipo_instancia']:<12} | "
                f"beneficio={float(r['totalBenefit']):.2f} | "
                f"distancia={float(r['totalDistanceKm']):.2f} km | "
                f"tiempo={float(r['elapsedSeconds']):.2f} s"
            )

        print()
        print("=" * 120)
        print("TABLA COMPARATIVA DE ALFAS - MEDIAS")
        print("=" * 120)
        for _, r in ahorro_mean_by_alpha_df.iterrows():
            print(
                f"alfa={format_alpha(r['alfa']):<6} | "
                f"n={int(r['n_instancias'])} | "
                f"beneficio medio={float(r['beneficio_medio']):.2f} | "
                f"distancia media={float(r['distancia_media_km']):.2f} km | "
                f"tiempo medio={float(r['tiempo_medio_s']):.2f} s | "
                f"carreras medias={float(r['n_carreras_medio']):.2f}"
            )
    else:
        print()
        print("No se han podido generar tablas de alfas porque no hay soluciones válidas del modo ahorro.")

    print()
    print("=" * 120)
    print("FICHEROS COMPARATIVOS GENERADOS")
    print("=" * 120)
    print(OUTPUT_DIR / "tabla_comparativa_mejores_soluciones.csv")
    print(OUTPUT_DIR / "tabla_comparativa_medias.csv")
    print(OUTPUT_DIR / "tabla_comparativa_medias_por_tipo.csv")
    print(OUTPUT_DIR / "tabla_comparativa_alfas_mejores.csv")
    print(OUTPUT_DIR / "tabla_comparativa_alfas_medias.csv")
    print(OUTPUT_DIR / "tabla_comparativa_alfas_medias_por_tipo.csv")


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    args = parse_args()

    modes = parse_modes(args.mode)
    alphas = parse_alphas(args.alfa)
    instances_root = Path(args.instances_root)
    full_data_file = Path(args.full_data_file)

    first_circuit_ref = normalize_optional_arg(args.first_circuit)
    last_circuit_ref = normalize_optional_arg(args.last_circuit)

    run_small = args.small
    run_medium = args.medium
    run_total_data = args.total_data

    # Mantener comportamiento anterior:
    # si no se indica nada, ejecutar pequeñas y medias
    if not run_small and not run_medium and not run_total_data:
        run_small = True
        run_medium = True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    instance_files = get_instance_files(
        instances_root,
        run_small,
        run_medium,
        run_total_data,
        full_data_file,
    )

    if not instance_files:
        raise FileNotFoundError(
            "No se encontraron ficheros para ejecutar. "
            f"Root={instances_root.resolve()} | full={full_data_file.resolve()}"
        )

    print("=" * 120)
    print("LANZADOR DE EXPERIMENTOS")
    print("=" * 120)
    print(f"Modos seleccionados: {modes}")
    print(f"Alfas seleccionados para ahorro: {[format_alpha(a) for a in alphas]}")
    print(f"Instancias encontradas: {len(instance_files)}")
    print(f"Ejecutar pequeñas: {run_small}")
    print(f"Ejecutar medias: {run_medium}")
    print(f"Ejecutar fichero completo: {run_total_data}")
    print(f"Fichero completo: {full_data_file}")
    print(f"Circuito inicial fijo: {first_circuit_ref}")
    print(f"Circuito final fijo: {last_circuit_ref}")

    all_summary_dfs: List[pd.DataFrame] = []
    all_successful_records: List[Dict[str, Any]] = []

    for mode in modes:
        summary_df, _, successful_records = process_mode(
            mode,
            instance_files,
            first_circuit_ref,
            last_circuit_ref,
            alphas,
            full_data_file,
        )
        all_summary_dfs.append(summary_df)
        all_successful_records.extend(successful_records)

    save_global_outputs(all_summary_dfs, all_successful_records)
    save_comparison_tables(all_summary_dfs)


if __name__ == "__main__":
    main()