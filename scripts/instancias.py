#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
import re
import pandas as pd
from pathlib import Path
from typing import Set



SMALL_SIZE = 35
MEDIUM_SIZE = 50


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"[^a-z0-9]+", "", c.lower().strip()) for c in df.columns]
    return df


def find_col(df: pd.DataFrame, *candidates: str) -> str:
    normalized = [re.sub(r"[^a-z0-9]+", "", c.lower().strip()) for c in candidates]
    for cand in normalized:
        if cand in df.columns:
            return cand
    raise ValueError(
        f"No se encontró ninguna de estas columnas: {candidates}. "
        f"Columnas disponibles: {list(df.columns)}"
    )


def parse_unavailable_weeks(value) -> Set[int]:
    if pd.isna(value):
        return set()
    text = str(value).strip()
    if not text:
        return set()

    weeks = set()
    for token in text.split(","):
        token = token.strip()
        if token:
            weeks.add(int(token))
    return weeks


def available_in_week(unavailable: Set[int], week: int) -> bool:
    return week not in unavailable


def validate_fixed_circuits(df: pd.DataFrame, circuit_id_col: str, unavailable_col: str,
                            origin: str, destination: str) -> None:
    ids = set(df[circuit_id_col].astype(str))

    if origin not in ids:
        raise ValueError(f"El circuito de origen '{origin}' no existe en el CSV.")
    if destination not in ids:
        raise ValueError(f"El circuito de destino '{destination}' no existe en el CSV.")
    if origin == destination:
        raise ValueError("Origen y destino no pueden ser el mismo circuito.")

    origin_row = df[df[circuit_id_col].astype(str) == origin].iloc[0]
    dest_row = df[df[circuit_id_col].astype(str) == destination].iloc[0]

    origin_unavailable = parse_unavailable_weeks(origin_row[unavailable_col])
    dest_unavailable = parse_unavailable_weeks(dest_row[unavailable_col])

    if not available_in_week(origin_unavailable, 1):
        raise ValueError(
            f"El circuito origen '{origin}' no está disponible en la semana 1."
        )

    if not available_in_week(dest_unavailable, 52):
        raise ValueError(
            f"El circuito destino '{destination}' no está disponible en la semana 52."
        )


def build_instance(df: pd.DataFrame, circuit_id_col: str, size: int,
                   origin: str, destination: str, rng: random.Random) -> pd.DataFrame:
    if size > len(df):
        raise ValueError(
            f"No se puede construir una instancia de tamaño {size} con solo {len(df)} circuitos."
        )

    all_ids = df[circuit_id_col].astype(str).tolist()
    pool = [cid for cid in all_ids if cid not in {origin, destination}]

    sampled = rng.sample(pool, size - 2)
    selected_ids = {origin, destination, *sampled}

    out = df[df[circuit_id_col].astype(str).isin(selected_ids)].copy()

    out["_fixed_order"] = out[circuit_id_col].astype(str).map(
        lambda x: 0 if x == origin else (2 if x == destination else 1)
    )
    out = out.sort_values(by=["_fixed_order", circuit_id_col]).drop(columns="_fixed_order")
    return out


def save_instances(df: pd.DataFrame, circuit_id_col: str, count: int, size: int,
                   prefix: str, origin: str, destination: str,
                   seed_base: int, out_dir: Path) -> list[dict]:
    summaries = []
    group_dir = out_dir / prefix.lower()
    group_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, count + 1):
        seed = seed_base + i
        rng = random.Random(seed)

        instance_df = build_instance(
            df=df,
            circuit_id_col=circuit_id_col,
            size=size,
            origin=origin,
            destination=destination,
            rng=rng
        )

        name = f"Instancia-{prefix}{i}"
        file_path = group_dir / f"{name}.csv"
        instance_df.to_csv(file_path, index=False, encoding="utf-8")

        summaries.append({
            "instancia": name,
            "tipo": "pequena" if size == SMALL_SIZE else "media",
            "tamano": size,
            "origen_fijo": origin,
            "destino_fijo": destination,
            "seed": seed,
            "archivo": str(file_path),
            "circuitos": ",".join(instance_df[circuit_id_col].astype(str).tolist())
        })

    return summaries


def main():
    parser = argparse.ArgumentParser(
        description="Genera instancias pequeñas (35) y medias (50) para el TFG con origen y destino fijos."
    )

    parser.add_argument(
        "-i", "--input",
        default="circuitos_f1_beneficios_disponible_simulados.csv",
        help="Ruta al CSV de circuitos."
    )
    parser.add_argument(
        "-o", "--output",
        default="instancias_generadas",
        help="Directorio de salida."
    )
    parser.add_argument(
        "-s", "--small",
        type=int,
        default=0,
        help="Número de instancias pequeñas a generar (35 circuitos)."
    )
    parser.add_argument(
        "-m", "--medium",
        type=int,
        default=0,
        help="Número de instancias medias a generar (50 circuitos)."
    )
    parser.add_argument(
        "--origen",
        required=True,
        help="circuitId del circuito fijo de inicio."
    )
    parser.add_argument(
        "--destino",
        required=True,
        help="circuitId del circuito fijo de final."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla base para reproducibilidad."
    )

    args = parser.parse_args()

    if args.small < 0 or args.medium < 0:
        raise ValueError("Los valores de -s y -m deben ser >= 0.")

    if args.small == 0 and args.medium == 0:
        raise ValueError("Debes pedir al menos una instancia con -s o -m.")

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(input_path)
    df = normalize_columns(df_raw)

    circuit_id_col = find_col(df, "circuitId", "circuit_id", "id")
    unavailable_col = find_col(df, "semanasNoDisponibles", "semanas_no_disponibles", "unavailableWeeks")

    validate_fixed_circuits(
        df=df,
        circuit_id_col=circuit_id_col,
        unavailable_col=unavailable_col,
        origin=args.origen,
        destination=args.destino
    )

    all_summaries = []

    if args.small > 0:
        all_summaries.extend(
            save_instances(
                df=df,
                circuit_id_col=circuit_id_col,
                count=args.small,
                size=SMALL_SIZE,
                prefix="S",
                origin=args.origen,
                destination=args.destino,
                seed_base=args.seed + 1000,
                out_dir=output_dir
            )
        )

    if args.medium > 0:
        all_summaries.extend(
            save_instances(
                df=df,
                circuit_id_col=circuit_id_col,
                count=args.medium,
                size=MEDIUM_SIZE,
                prefix="M",
                origin=args.origen,
                destination=args.destino,
                seed_base=args.seed + 2000,
                out_dir=output_dir
            )
        )

    summary_df = pd.DataFrame(all_summaries)
    summary_path = output_dir / "resumen_instancias.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    print("Instancias generadas correctamente.")
    print(f"Resumen guardado en: {summary_path}")
    print(summary_df[["instancia", "tipo", "tamano", "origen_fijo", "destino_fijo", "seed"]].to_string(index=False))


if __name__ == "__main__":
    main()