from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

import VNS_FINAL as vns


# ============================================================
# CONFIGURACIÓN POR DEFECTO
# ============================================================
DEFAULT_INSTANCES_ROOT = Path("instancias_generadas")
DEFAULT_FULL_DATA_FILE = Path("circuitos_f1_beneficios_disponible_simulados.csv")
DEFAULT_OUTPUT_DIR = Path("resultados_experimentos_vns")
DEFAULT_TMP_DIRNAME = "_tmp_normalized"

DEFAULT_FIRST_CIRCUIT = "avus"
DEFAULT_LAST_CIRCUIT = "jarama"
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_LS1_CANDIDATOS = [150]


# ============================================================
# ARGUMENTOS DE TERMINAL
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta experimentos del VNS sobre instancias pequeñas, medias y/o "
            "el fichero completo, permitiendo múltiples seeds y múltiples valores "
            "de LS1_MAX_CANDIDATOS."
        )
    )

    parser.add_argument("-s", "--small", action="store_true", help="Ejecutar instancias pequeñas")
    parser.add_argument("-m", "--medium", action="store_true", help="Ejecutar instancias medias")
    parser.add_argument(
        "-td",
        "--total-data",
        action="store_true",
        help="Ejecutar también el archivo completo de datos",
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
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Carpeta donde se guardarán los resultados",
    )

    parser.add_argument(
        "--first-circuit",
        type=str,
        default=DEFAULT_FIRST_CIRCUIT,
        help="Circuito fijo inicial (circuitId o nombre).",
    )
    parser.add_argument(
        "--last-circuit",
        type=str,
        default=DEFAULT_LAST_CIRCUIT,
        help="Circuito fijo final (circuitId o nombre).",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        default=[str(x) for x in DEFAULT_SEEDS],
        help="Lista de seeds. Ej: --seeds 1 2 3 o --seeds 1,2,3",
    )

    parser.add_argument(
        "--ls1-max-candidatos",
        nargs="+",
        default=[str(x) for x in DEFAULT_LS1_CANDIDATOS],
        help="Lista de valores para LS1_MAX_CANDIDATOS. Ej: --ls1-max-candidatos 50 100 200",
    )

    parser.add_argument(
        "--alphas",
        nargs="+",
        default=None,
        help="Lista de alphas para la construcción inicial. Ej: --alphas 0.1 0.3 0.5",
    )
    parser.add_argument("--max-time", type=float, default=None, help="Sobrescribe MAX_TIME_SEC")
    parser.add_argument("--k-max", type=int, default=None, help="Sobrescribe K_MAX")
    parser.add_argument("--beta", type=float, default=None, help="Sobrescribe BETA")
    parser.add_argument("--t0-sa", type=float, default=None, help="Sobrescribe T0_SA")
    parser.add_argument("--lambda-sa", type=float, default=None, help="Sobrescribe LAMBDA_SA")
    parser.add_argument("--max-races", type=int, default=None, help="Sobrescribe NUM_CARRERAS")
    parser.add_argument(
        "--max-consecutive",
        type=int,
        default=None,
        help="Sobrescribe MAX_FINES_SEGUIDOS",
    )
    parser.add_argument("--t-max", type=float, default=None, help="Sobrescribe T_MAX")
    parser.add_argument("--velocity", type=float, default=None, help="Sobrescribe VEL")
    parser.add_argument("--weeks", type=int, default=None, help="Sobrescribe NUM_SEMANAS")
    parser.add_argument("--debug-vns", action="store_true", help="Activa DEBUG_VNS")
    parser.add_argument("--debug-ls1", action="store_true", help="Activa DEBUG_LS1")

    return parser.parse_args()


def parse_int_list(raw_values: List[str]) -> List[int]:
    parsed: List[int] = []
    for item in raw_values:
        for token in str(item).split(","):
            token = token.strip()
            if not token:
                continue
            value = int(token)
            if value not in parsed:
                parsed.append(value)

    if not parsed:
        raise ValueError("Debes proporcionar al menos un entero válido.")
    return parsed


def parse_float_list(raw_values: List[str] | None) -> List[float] | None:
    if raw_values is None:
        return None

    parsed: List[float] = []
    for item in raw_values:
        for token in str(item).split(","):
            token = token.strip()
            if not token:
                continue
            value = float(token)
            if value not in parsed:
                parsed.append(value)

    if not parsed:
        return None
    return parsed


def normalize_optional_arg(value: str | None) -> str | None:
    if value is None:
        return None
    if str(value).strip().lower() in {"none", "null", ""}:
        return None
    return value


def format_alpha(alpha: float | None) -> str:
    if alpha is None:
        return "NA"
    return f"{alpha:.6f}".rstrip("0").rstrip(".")


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


def validate_columns_for_vns(df: pd.DataFrame, input_name: str) -> None:
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
            f"[VNS] Faltan columnas requeridas {missing} en {input_name}. "
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
    # Si se activa -td, ejecutar SOLO el fichero completo
    if run_total_data:
        if full_data_file is None:
            raise ValueError("Debes indicar --full-data-file cuando uses -td.")
        if not full_data_file.exists():
            raise FileNotFoundError(
                f"No se encontró el fichero completo: {full_data_file.resolve()}"
            )
        return [full_data_file]

    files: List[Path] = []

    # Si no se especifica ni -s ni -m, ejecutar pequeñas e intermedias por defecto
    if not run_small and not run_medium:
        run_small = True
        run_medium = True

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

    if not files:
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


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371.0
    lat1_r = math.radians(lat1)
    lon1_r = math.radians(lon1)
    lat2_r = math.radians(lat2)
    lon2_r = math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def print_final_calendar(calendar_df: pd.DataFrame) -> None:
    print()
    print("-" * 120)

    if calendar_df.empty:
        print("No hay calendario que mostrar.")
        print("-" * 120)
        return

    cols_to_show = ["Semana", "CircuitId", "Circuito", "Beneficio_M_USD"]
    cal = calendar_df[cols_to_show].copy()

    cal["Semana"] = cal["Semana"].astype(int)
    cal["Beneficio_M_USD"] = cal["Beneficio_M_USD"].astype(float)

    print(cal.to_string(index=True))
    print("-" * 120)
    print(f"{'TOTAL':<46}{cal['Beneficio_M_USD'].sum():>20.2f}")
    print("=" * 120)


# ============================================================
# CONFIGURACIÓN DEL MÓDULO VNS
# ============================================================
def configure_vns_module(
    normalized_csv: Path,
    first_name: str,
    last_name: str,
    seed: int,
    ls1_max_candidatos: int,
    args: argparse.Namespace,
) -> None:
    vns.RUTA_CSV_CIRCUITOS = str(normalized_csv)
    vns.NOMBRE_CIRCUITO_ORIGEN = first_name
    vns.NOMBRE_CIRCUITO_DESTINO = last_name
    vns.RANDOM_SEED = seed
    vns.LS1_MAX_CANDIDATOS = ls1_max_candidatos

    if args.alphas is not None:
        vns.ALPHAS = list(args.alphas)
    if args.max_time is not None:
        vns.MAX_TIME_SEC = args.max_time
    if args.k_max is not None:
        vns.K_MAX = args.k_max
    if args.beta is not None:
        vns.BETA = args.beta
    if args.t0_sa is not None:
        vns.T0_SA = args.t0_sa
    if args.lambda_sa is not None:
        vns.LAMBDA_SA = args.lambda_sa
    if args.max_races is not None:
        vns.NUM_CARRERAS = args.max_races
    if args.max_consecutive is not None:
        vns.MAX_FINES_SEGUIDOS = args.max_consecutive
    if args.t_max is not None:
        vns.T_MAX = args.t_max
    if args.velocity is not None:
        vns.VEL = args.velocity
    if args.weeks is not None:
        vns.NUM_SEMANAS = args.weeks

    vns.DEBUG_VNS = bool(args.debug_vns)
    vns.DEBUG_LS1 = bool(args.debug_ls1)


# ============================================================
# CONSTRUCCIÓN DE SALIDAS DE VNS
# ============================================================
def build_vns_outputs(
    df: pd.DataFrame,
    datos: Any,
    mejor_ruta: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    calendario = vns.crearCalendarioDesdeEstado(mejor_ruta)

    name_to_id = dict(zip(df["nombre"].astype(str), df["circuitId"].astype(str)))
    name_to_benefit = dict(
        zip(df["nombre"].astype(str), df["beneficio_total_millones_usd"].astype(float))
    )

    calendar_rows = []
    travel_rows = []
    total_distance_km = 0.0

    semanas_ordenadas = sorted(calendario.keys())
    for idx, semana in enumerate(semanas_ordenadas):
        nodo_id = calendario[semana]
        nodo = datos.nodos[nodo_id]

        circuito_name = nodo.circuitoBaseNombre or nodo.nombre
        circuito_id = name_to_id.get(circuito_name, circuito_name)
        benefit = float(name_to_benefit.get(circuito_name, nodo.recompensa))

        calendar_rows.append(
            {
                "Semana": int(semana),
                "CircuitId": circuito_id,
                "Circuito": circuito_name,
                "Ciudad": nodo.ciudad,
                "Pais": nodo.pais,
                "Beneficio_M_USD": benefit,
            }
        )

        if idx > 0:
            prev_week = semanas_ordenadas[idx - 1]
            prev_nodo_id = calendario[prev_week]
            prev_nodo = datos.nodos[prev_nodo_id]

            prev_name = prev_nodo.circuitoBaseNombre or prev_nodo.nombre
            prev_id = name_to_id.get(prev_name, prev_name)

            dist_km = haversine_km(prev_nodo.x, prev_nodo.y, nodo.x, nodo.y)
            total_distance_km += dist_km

            travel_rows.append(
                {
                    "Semana_origen": int(prev_week),
                    "CircuitoId_origen": prev_id,
                    "Circuito_origen": prev_name,
                    "Semana_destino": int(semana),
                    "CircuitoId_destino": circuito_id,
                    "Circuito_destino": circuito_name,
                    "Distancia_km": float(dist_km),
                }
            )

    calendar_df = pd.DataFrame(calendar_rows)
    travel_df = pd.DataFrame(travel_rows)

    total_benefit_exact = (
        float(calendar_df["Beneficio_M_USD"].sum()) if not calendar_df.empty else 0.0
    )

    route_distance_with_depots = (
        float(mejor_ruta.tiempoTotal) * float(getattr(vns, "VEL", 1.0))
        if float(getattr(vns, "VEL", 1.0)) != 0.0
        else float(mejor_ruta.tiempoTotal)
    )

    results: Dict[str, Any] = {
        "status": "HEURISTIC_OK",
        "objectiveValue": float(vns.beneficioRuta(mejor_ruta)),
        "selectedCircuits": calendar_df["CircuitId"].tolist() if not calendar_df.empty else [],
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
        "routeDistanceWithDepotsKm": float(route_distance_with_depots),
        "nRaces": int(len(calendar_df)),
    }

    return calendar_df, travel_df, results


# ============================================================
# RUNNER VNS
# ============================================================
def run_vns_instance(
    csv_file: Path,
    tmp_dir: Path,
    first_circuit_ref: str | None,
    last_circuit_ref: str | None,
    seed: int,
    ls1_max_candidatos: int,
    args: argparse.Namespace,
    full_data_file: Path | None,
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    normalized_csv = tmp_dir / csv_file.name
    normalized_csv = standardize_instance_csv(csv_file, normalized_csv)

    df = pd.read_csv(normalized_csv)
    validate_columns_for_vns(df, csv_file.name)

    first_id = resolve_circuit_id(df, first_circuit_ref)
    last_id = resolve_circuit_id(df, last_circuit_ref)
    first_name = resolve_circuit_name(df, first_circuit_ref)
    last_name = resolve_circuit_name(df, last_circuit_ref)

    if first_name is None or last_name is None:
        raise ValueError(
            "El VNS actual necesita circuito inicial y final fijos. "
            "Indica --first-circuit y --last-circuit."
        )

    configure_vns_module(
        normalized_csv=normalized_csv,
        first_name=first_name,
        last_name=last_name,
        seed=seed,
        ls1_max_candidatos=ls1_max_candidatos,
        args=args,
    )

    t0 = time.perf_counter()
    datos = vns.prepararDatosProblema()
    mejor_ruta, alpha_sel = vns.vnsDeterminista(datos)
    elapsed = time.perf_counter() - t0

    calendar_df, travel_df, results = build_vns_outputs(df, datos, mejor_ruta)

    run_name = f"{csv_file.stem}_vns_seed{seed}_ls1c{ls1_max_candidatos}"

    row: Dict[str, Any] = {
        "mode": "vns",
        "seed": seed,
        "ls1_max_candidatos": ls1_max_candidatos,
        "alfa_seleccionado": alpha_sel,
        "alphas_usados": ",".join(format_alpha(a) for a in getattr(vns, "ALPHAS", [])),
        "config_label": f"vns_seed{seed}_ls1c{ls1_max_candidatos}",
        "instancia": csv_file.stem,
        "run_name": run_name,
        "fichero_original": str(csv_file),
        "fichero_normalizado": str(normalized_csv),
        "tipo_instancia": classify_instance(csv_file, full_data_file),
        "status": str(results.get("status")),
        "objectiveValue": results.get("objectiveValue"),
        "totalBenefit": results.get("totalBenefit"),
        "nRaces": results.get("nRaces"),
        "totalDistanceKm": results.get("totalDistanceKm"),
        "routeDistanceWithDepotsKm": results.get("routeDistanceWithDepotsKm"),
        "elapsedSeconds": elapsed,
        "firstCircuit": first_id,
        "lastCircuit": last_id,
        "maxTimeSec": getattr(vns, "MAX_TIME_SEC", None),
        "kMax": getattr(vns, "K_MAX", None),
        "beta": getattr(vns, "BETA", None),
    }

    return row, results, calendar_df, travel_df


# ============================================================
# PROCESADO DEL MODO VNS
# ============================================================
def process_vns(
    instance_files: List[Path],
    seeds: List[int],
    ls1_values: List[int],
    first_circuit_ref: str | None,
    last_circuit_ref: str | None,
    args: argparse.Namespace,
    output_dir: Path,
    full_data_file: Path | None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    mode_output_dir = output_dir / "vns"
    mode_tmp_dir = output_dir / DEFAULT_TMP_DIRNAME / "vns"

    mode_output_dir.mkdir(parents=True, exist_ok=True)
    mode_tmp_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    successful_records: List[Dict[str, Any]] = []

    runs: List[Tuple[Path, int, int]] = []
    for csv_file in instance_files:
        for seed in seeds:
            for ls1_value in ls1_values:
                runs.append((csv_file, seed, ls1_value))

    print()
    print("=" * 120)
    print("EJECUCIÓN DEL MODO: VNS")
    print("=" * 120)

    for i, (csv_file, seed, ls1_value) in enumerate(runs, start=1):
        print("-" * 120)
        print(
            f"[{i}/{len(runs)}] [vns] Ejecutando: {csv_file.name} | "
            f"seed={seed} | ls1_max_candidatos={ls1_value}"
        )

        try:
            row, results, calendar_df, travel_df = run_vns_instance(
                csv_file=csv_file,
                tmp_dir=mode_tmp_dir,
                first_circuit_ref=first_circuit_ref,
                last_circuit_ref=last_circuit_ref,
                seed=seed,
                ls1_max_candidatos=ls1_value,
                args=args,
                full_data_file=full_data_file,
            )

            summary_rows.append(row)

            if results.get("objectiveValue") is not None:
                save_details(calendar_df, travel_df, mode_output_dir, row["run_name"])

                successful_records.append(
                    {
                        "row": row,
                        "results": results,
                        "calendar_df": calendar_df,
                        "travel_df": travel_df,
                    }
                )

                print(
                    f"OK | seed={seed} | ls1={ls1_value} | "
                    f"alfa_sel={format_alpha(row['alfa_seleccionado'])} | "
                    f"beneficio={row['totalBenefit']:.2f} M USD | "
                    f"distancia={row['totalDistanceKm']:.2f} km | "
                    f"tiempo={row['elapsedSeconds']:.2f} s | "
                    f"carreras={row['nRaces']}"
                )
            else:
                print(f"SIN SOLUCIÓN | status={row['status']}")

        except Exception as e:
            error_row = {
                "mode": "vns",
                "seed": seed,
                "ls1_max_candidatos": ls1_value,
                "alfa_seleccionado": None,
                "alphas_usados": None,
                "config_label": f"vns_seed{seed}_ls1c{ls1_value}",
                "instancia": csv_file.stem,
                "run_name": f"{csv_file.stem}_vns_seed{seed}_ls1c{ls1_value}",
                "fichero_original": str(csv_file),
                "fichero_normalizado": None,
                "tipo_instancia": classify_instance(csv_file, full_data_file),
                "status": f"ERROR: {e}",
                "objectiveValue": None,
                "totalBenefit": None,
                "nRaces": None,
                "totalDistanceKm": None,
                "routeDistanceWithDepotsKm": None,
                "elapsedSeconds": None,
                "firstCircuit": first_circuit_ref,
                "lastCircuit": last_circuit_ref,
                "maxTimeSec": getattr(vns, "MAX_TIME_SEC", None),
                "kMax": getattr(vns, "K_MAX", None),
                "beta": getattr(vns, "BETA", None),
            }
            summary_rows.append(error_row)
            print(f"ERROR en {csv_file.name} | seed={seed} | ls1={ls1_value}: {e}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        mode_output_dir / "resumen_ejecuciones_vns.csv",
        index=False,
        encoding="utf-8",
    )

    ok_df = summary_df[summary_df["objectiveValue"].notna()].copy()
    if not ok_df.empty:
        ok_df = ok_df.sort_values(
            ["totalBenefit", "totalDistanceKm", "elapsedSeconds"],
            ascending=[False, True, True],
        )

    ok_df.to_csv(
        mode_output_dir / "ranking_resultados_vns.csv",
        index=False,
        encoding="utf-8",
    )

    if not ok_df.empty:
        list_cols = [
            "mode",
            "seed",
            "ls1_max_candidatos",
            "alfa_seleccionado",
            "config_label",
            "instancia",
            "tipo_instancia",
            "totalBenefit",
            "totalDistanceKm",
            "routeDistanceWithDepotsKm",
            "elapsedSeconds",
            "nRaces",
            "status",
        ]
        list_df = ok_df[list_cols].copy()
        list_df.to_csv(
            mode_output_dir / "lista_beneficios_distancias_tiempos_vns.csv",
            index=False,
            encoding="utf-8",
        )

        stats_rows = [build_stats("Global", ok_df)]
        for group_name, group_df in ok_df.groupby("tipo_instancia"):
            stats_rows.append(build_stats(group_name, group_df))
        for seed, seed_df in ok_df.groupby("seed"):
            stats_rows.append(build_stats(f"seed={seed}", seed_df))
        for ls1_value, ls1_df in ok_df.groupby("ls1_max_candidatos"):
            stats_rows.append(build_stats(f"ls1={ls1_value}", ls1_df))

        stats_df = pd.DataFrame(stats_rows)
        stats_df.to_csv(
            mode_output_dir / "estadisticas_medias_vns.csv",
            index=False,
            encoding="utf-8",
        )

        best_row = select_best_row(ok_df)
        best_record = next(
            rec for rec in successful_records
            if rec["row"]["run_name"] == best_row["run_name"]
        )

        best_record["calendar_df"].to_csv(
            mode_output_dir / "mejor_calendario_vns.csv",
            index=False,
            encoding="utf-8",
        )
        best_record["travel_df"].to_csv(
            mode_output_dir / "mejor_viaje_vns.csv",
            index=False,
            encoding="utf-8",
        )

        best_by_instance = (
            ok_df.sort_values(
                ["totalBenefit", "totalDistanceKm", "elapsedSeconds"],
                ascending=[False, True, True],
            )
            .drop_duplicates(subset=["instancia"], keep="first")
            .copy()
        )
        best_by_instance.to_csv(
            mode_output_dir / "mejor_config_por_instancia_vns.csv",
            index=False,
            encoding="utf-8",
        )

        mean_by_instance = (
            ok_df.groupby(["instancia", "tipo_instancia"], as_index=False)
            .agg(
                n_runs=("instancia", "size"),
                beneficio_medio=("totalBenefit", "mean"),
                beneficio_std=("totalBenefit", lambda x: x.std(ddof=0)),
                distancia_media_km=("totalDistanceKm", "mean"),
                distancia_std_km=("totalDistanceKm", lambda x: x.std(ddof=0)),
                tiempo_medio_s=("elapsedSeconds", "mean"),
                tiempo_std_s=("elapsedSeconds", lambda x: x.std(ddof=0)),
                carreras_medias=("nRaces", "mean"),
                carreras_std=("nRaces", lambda x: x.std(ddof=0)),
            )
            .sort_values(
                ["beneficio_medio", "distancia_media_km", "tiempo_medio_s"],
                ascending=[False, True, True],
            )
        )
        mean_by_instance.to_csv(
            mode_output_dir / "medias_por_instancia_vns.csv",
            index=False,
            encoding="utf-8",
        )

        print()
        print("=" * 120)
        print("LISTA DE RESULTADOS - VNS")
        print("=" * 120)
        for _, r in list_df.iterrows():
            print(
                f"{r['instancia']:<18} | {r['tipo_instancia']:<12} | "
                f"seed={int(r['seed']):<4} | "
                f"ls1={int(r['ls1_max_candidatos']):<4} | "
                f"alfa_sel={format_alpha(r['alfa_seleccionado']):<6} | "
                f"beneficio={float(r['totalBenefit']):.2f} M USD | "
                f"distancia={float(r['totalDistanceKm']):.2f} km | "
                f"tiempo={float(r['elapsedSeconds']):.2f} s | "
                f"carreras={int(r['nRaces'])}"
            )

        print()
        print("=" * 120)
        print("ESTADÍSTICAS MEDIAS - VNS")
        print("=" * 120)
        for _, r in stats_df.iterrows():
            print(
                f"{r['grupo']:<18} | "
                f"n={int(r['n_instancias'])} | "
                f"beneficio medio={float(r['beneficio_medio']):.2f} M USD | "
                f"beneficio std={float(r['beneficio_std']):.2f} | "
                f"distancia media={float(r['distancia_media_km']):.2f} km | "
                f"distancia std={float(r['distancia_std_km']):.2f} | "
                f"tiempo medio={float(r['tiempo_medio_s']):.2f} s | "
                f"tiempo std={float(r['tiempo_std_s']):.2f} | "
                f"carreras medias={float(r['n_carreras_medio']):.2f} | "
                f"carreras std={float(r['n_carreras_std']):.2f}"
            )

        print()
        print("=" * 120)
        print("MEJOR RESULTADO - VNS")
        print("=" * 120)
        print(f"Instancia: {best_row['instancia']}")
        print(f"Tipo: {best_row['tipo_instancia']}")
        print(f"Seed: {int(best_row['seed'])}")
        print(f"LS1_MAX_CANDIDATOS: {int(best_row['ls1_max_candidatos'])}")
        print(f"Alpha seleccionado: {format_alpha(best_row['alfa_seleccionado'])}")
        print(f"Beneficio total: {float(best_row['totalBenefit']):.2f} M USD")
        print(f"Distancia total: {float(best_row['totalDistanceKm']):.2f} km")
        print(f"Tiempo: {float(best_row['elapsedSeconds']):.2f} s")
        print(f"Número de carreras: {int(best_row['nRaces'])}")

        print_final_calendar(best_record["calendar_df"])
    else:
        print("\nNo hubo soluciones válidas para el modo VNS.")

    return summary_df, ok_df, successful_records


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    args = parse_args()

    args.seeds = parse_int_list(args.seeds)
    args.ls1_max_candidatos = parse_int_list(args.ls1_max_candidatos)
    args.alphas = parse_float_list(args.alphas)

    instances_root = Path(args.instances_root)
    full_data_file = Path(args.full_data_file) if args.full_data_file else None
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / DEFAULT_TMP_DIRNAME).mkdir(parents=True, exist_ok=True)

    first_circuit_ref = normalize_optional_arg(args.first_circuit)
    last_circuit_ref = normalize_optional_arg(args.last_circuit)

    if first_circuit_ref is None or last_circuit_ref is None:
        raise ValueError(
            "Con tu VNS actual debes fijar circuito inicial y final. "
            "No uses 'none' en --first-circuit / --last-circuit."
        )

    instance_files = get_instance_files(
        root=instances_root,
        run_small=bool(args.small),
        run_medium=bool(args.medium),
        run_total_data=bool(args.total_data),
        full_data_file=full_data_file,
    )

    if not instance_files:
        raise FileNotFoundError("No se han encontrado instancias CSV para ejecutar.")

    ls1_values = args.ls1_max_candidatos or DEFAULT_LS1_CANDIDATOS

    print("=" * 120)
    print("CONFIGURACIÓN DE LA EJECUCIÓN VNS")
    print("=" * 120)
    print(f"Instancias: {len(instance_files)}")
    print(f"Seeds: {args.seeds}")
    print(f"LS1_MAX_CANDIDATOS: {ls1_values}")
    print(f"Carpeta instancias: {instances_root.resolve()}")
    print(f"Fichero completo: {full_data_file.resolve() if full_data_file else 'NA'}")
    print(f"Circuito inicial: {first_circuit_ref}")
    print(f"Circuito final: {last_circuit_ref}")
    print(f"Output dir: {output_dir.resolve()}")
    if args.alphas is not None:
        print(f"Alphas forzados: {args.alphas}")
    if args.max_time is not None:
        print(f"MAX_TIME_SEC: {args.max_time}")
    if args.k_max is not None:
        print(f"K_MAX: {args.k_max}")
    if args.beta is not None:
        print(f"BETA: {args.beta}")

    summary_df, ok_df, _ = process_vns(
        instance_files=instance_files,
        seeds=args.seeds,
        ls1_values=ls1_values,
        first_circuit_ref=first_circuit_ref,
        last_circuit_ref=last_circuit_ref,
        args=args,
        output_dir=output_dir,
        full_data_file=full_data_file,
    )

    print()
    print("=" * 120)
    print("FICHEROS GENERADOS")
    print("=" * 120)
    print(output_dir / "vns" / "resumen_ejecuciones_vns.csv")
    print(output_dir / "vns" / "ranking_resultados_vns.csv")
    print(output_dir / "vns" / "lista_beneficios_distancias_tiempos_vns.csv")
    print(output_dir / "vns" / "estadisticas_medias_vns.csv")
    print(output_dir / "vns" / "mejor_calendario_vns.csv")
    print(output_dir / "vns" / "mejor_viaje_vns.csv")
    print(output_dir / "vns" / "mejor_config_por_instancia_vns.csv")
    print(output_dir / "vns" / "medias_por_instancia_vns.csv")

    if not ok_df.empty:
        best_row = select_best_row(ok_df)
        print()
        print("=" * 120)
        print("RESUMEN FINAL")
        print("=" * 120)
        print(f"Mejor run: {best_row['run_name']}")
        print(f"Instancia: {best_row['instancia']}")
        print(f"Tipo: {best_row['tipo_instancia']}")
        print(f"Seed: {int(best_row['seed'])}")
        print(f"LS1_MAX_CANDIDATOS: {int(best_row['ls1_max_candidatos'])}")
        print(f"Alpha seleccionado: {format_alpha(best_row['alfa_seleccionado'])}")
        print(f"Beneficio: {float(best_row['totalBenefit']):.2f} M USD")
        print(f"Distancia: {float(best_row['totalDistanceKm']):.2f} km")
        print(f"Tiempo: {float(best_row['elapsedSeconds']):.2f} s")


if __name__ == "__main__":
    main()