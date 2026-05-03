"""
Objetivo: Optimizar un calendario de 24 Grandes Premios maximizando beneficios,
respetando ventanas de disponibilidad de cada circuito y mostrando la distancia
total del calendario generado.
"""

import math
from typing import Dict, List, Tuple, Set, Any

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition


class F1CalendarOptimizer:
    def __init__(self, csvPath: str) -> None:
        self.df: pd.DataFrame = pd.read_csv(csvPath)
        self.circuits: List[str] = self.df["circuitId"].tolist()
        self.nCircuits: int = len(self.circuits)
        self.nWeeks: int = 52
        self.nRaces: int = 24

        self.benefits: Dict[str, float] = dict(
            zip(self.df["circuitId"], self.df["beneficio_total_millones_usd"])
        )

        self.unavailableWeeks: Dict[str, Set[int]] = self._parseUnavailableWeeks()

        # Detectar columnas de coordenadas automáticamente
        self.latColumn, self.lonColumn = self._detectCoordinateColumns()
        self.coordinates: Dict[str, Tuple[float, float]] = self._buildCoordinates()

        self.model: pyo.ConcreteModel | None = None
        self._solverResults: Any = None

    def _parseUnavailableWeeks(self) -> Dict[str, Set[int]]:
        unavailable: Dict[str, Set[int]] = {}

        for _, row in self.df.iterrows():
            circuitId: str = row["circuitId"]
            weeksStr = row["semanas_no_disponibles"]

            weeks: Set[int] = set()
            if isinstance(weeksStr, str):
                weeks = {
                    int(week.strip())
                    for week in weeksStr.split(",")
                    if week.strip()
                }

            unavailable[circuitId] = weeks

        return unavailable

    def _detectCoordinateColumns(self) -> Tuple[str, str]:
        lat_candidates = [
            "lat", "latitude", "latitud", "circuit_lat", "circuit_latitude"
        ]
        lon_candidates = [
            "lon", "lng", "long", "longitude", "longitud",
            "circuit_lon", "circuit_lng", "circuit_longitude"
        ]

        normalized_columns = {col.strip().lower(): col for col in self.df.columns}

        lat_col = next(
            (normalized_columns[c] for c in lat_candidates if c in normalized_columns),
            None
        )
        lon_col = next(
            (normalized_columns[c] for c in lon_candidates if c in normalized_columns),
            None
        )

        if lat_col is None or lon_col is None:
            raise ValueError(
                "No se encontraron columnas de coordenadas en el CSV.\n"
                "Necesitas una columna de latitud y otra de longitud.\n"
                f"Columnas disponibles: {list(self.df.columns)}"
            )

        return lat_col, lon_col

    def _buildCoordinates(self) -> Dict[str, Tuple[float, float]]:
        coordinates: Dict[str, Tuple[float, float]] = {}

        for _, row in self.df.iterrows():
            circuit = row["circuitId"]
            lat = row[self.latColumn]
            lon = row[self.lonColumn]

            if pd.isna(lat) or pd.isna(lon):
                raise ValueError(
                    f"Faltan coordenadas para el circuito '{circuit}'. "
                    f"Revisa las columnas '{self.latColumn}' y '{self.lonColumn}'."
                )

            coordinates[circuit] = (float(lat), float(lon))

        return coordinates

    @staticmethod
    def _haversineKm(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        r = 6371.0  # Radio medio de la Tierra en km

        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))
        return r * c

    def _calculateCalendarDistance(
        self, calendar: List[Tuple[int, str, float]]
    ) -> Tuple[float, List[Tuple[int, str, int, str, float]]]:
        totalDistanceKm = 0.0
        travelLegs: List[Tuple[int, str, int, str, float]] = []

        if len(calendar) < 2:
            return totalDistanceKm, travelLegs

        for i in range(len(calendar) - 1):
            week1, circuit1, _ = calendar[i]
            week2, circuit2, _ = calendar[i + 1]

            lat1, lon1 = self.coordinates[circuit1]
            lat2, lon2 = self.coordinates[circuit2]

            distanceKm = self._haversineKm(lat1, lon1, lat2, lon2)
            totalDistanceKm += distanceKm

            travelLegs.append((week1, circuit1, week2, circuit2, distanceKm))

        return totalDistanceKm, travelLegs

    def _createModel(self) -> None:
        model = pyo.ConcreteModel()

        # Sets
        model.CIRCUITS = pyo.Set(initialize=self.circuits)
        model.WEEKS = pyo.RangeSet(1, self.nWeeks)
        model.WINDOW_START = pyo.RangeSet(1, self.nWeeks - 3)
        model.UNAVAILABLE = pyo.Set(
            dimen=2,
            initialize=[
                (circuit, week)
                for circuit, weeks in self.unavailableWeeks.items()
                for week in weeks
            ],
        )

        # Parameters
        model.benefit = pyo.Param(
            model.CIRCUITS,
            initialize=self.benefits,
            within=pyo.Reals,
        )

        # Decision variables
        model.x = pyo.Var(model.CIRCUITS, domain=pyo.Binary)
        model.y = pyo.Var(model.CIRCUITS, model.WEEKS, domain=pyo.Binary)

        self.model = model

    def _addConstraints(
        self,
        firstCircuit: str | None,
        lastCircuit: str | None,
    ) -> None:
        if self.model is None:
            raise RuntimeError("Modelo no creado")

        model = self.model

        # Restricción 2: Total de circuitos seleccionados <= número máximo de carreras
        def totalRacesRule(m):
            return sum(m.x[circuit] for circuit in m.CIRCUITS) <= self.nRaces

        model.totalRaces = pyo.Constraint(rule=totalRacesRule)

        # Restricción 3: Si un circuito está seleccionado, debe asignarse a exactamente una semana
        def assignmentRule(m, circuit):
            return sum(m.y[circuit, week] for week in m.WEEKS) == m.x[circuit]

        model.assignment = pyo.Constraint(model.CIRCUITS, rule=assignmentRule)

        # Restricción 4: Respetar ventanas de no disponibilidad
        def unavailableRule(m, circuit, week):
            return m.y[circuit, week] == 0

        if len(list(model.UNAVAILABLE)) > 0:
            model.unavailable = pyo.Constraint(
                model.UNAVAILABLE,
                rule=unavailableRule,
            )

        # Restricción 5: Máximo una carrera por semana
        def oneRaceWeekRule(m, week):
            return sum(m.y[circuit, week] for circuit in m.CIRCUITS) <= 1

        model.oneRaceWeek = pyo.Constraint(model.WEEKS, rule=oneRaceWeekRule)

        # Restricción 6: Máximo 3 carreras en cualquier ventana de 4 semanas consecutivas
        def max3RacesIn4WeeksRule(m, startWeek):
            weeks = range(startWeek, startWeek + 4)
            return sum(
                m.y[circuit, week]
                for circuit in m.CIRCUITS
                for week in weeks
            ) <= 3

        model.max3RacesIn4Weeks = pyo.Constraint(
            model.WINDOW_START,
            rule=max3RacesIn4WeeksRule,
        )

        # Restricción 7: Circuito primero especificado (si aplica)
        if firstCircuit is not None and firstCircuit in self.circuits:
            model.firstCircuitSelected = pyo.Constraint(
                expr=model.x[firstCircuit] == 1
            )
            model.firstCircuitWeek1 = pyo.Constraint(
                expr=model.y[firstCircuit, 1] == 1
            )

        # Restricción 8: Circuito último especificado (si aplica)
        if lastCircuit is not None and lastCircuit in self.circuits:
            model.lastCircuitSelected = pyo.Constraint(
                expr=model.x[lastCircuit] == 1
            )
            model.lastCircuitWeek52 = pyo.Constraint(
                expr=model.y[lastCircuit, self.nWeeks] == 1
            )

    def _setObjective(self) -> None:
        if self.model is None:
            raise RuntimeError("Modelo no creado")

        model = self.model

        model.objective = pyo.Objective(
            expr=sum(
                model.benefit[circuit] * model.x[circuit]
                for circuit in model.CIRCUITS
            ),
            sense=pyo.maximize,
        )

    def optimize(
        self,
        firstCircuit: str | None = None,
        lastCircuit: str | None = None,
    ) -> Dict[str, Any]:
        print("=" * 80)
        print("F1 GRAND PRIX CALENDAR OPTIMIZATION")
        print("=" * 80)

        print("\n[1/5] Creando modelo de optimización...")
        self._createModel()

        print("[2/5] Añadiendo restricciones...")
        self._addConstraints(firstCircuit, lastCircuit)

        print("[3/5] Estableciendo función objetivo...")
        self._setObjective()

        print("[4/5] Resolviendo modelo...")
        if self.model is None:
            raise RuntimeError("Modelo no creado")

        solver = SolverFactory("gurobi")
        solverResults = solver.solve(self.model, tee=False)
        self._solverResults = solverResults

        print("[5/5] Procesando resultados...")
        results: Dict[str, Any] = self._processResults()

        return results

    def _processResults(self) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Modelo no creado o no resuelto")

        model = self.model
        termCond = getattr(getattr(self, "_solverResults", None), "solver", None)
        if termCond is not None:
            termCond = termCond.termination_condition

        optimal = termCond == TerminationCondition.optimal

        results: Dict[str, Any] = {
            "status": termCond,
            "objectiveValue": pyo.value(model.objective) if optimal else None,
            "selectedCircuits": [],
            "calendar": [],
            "travelLegs": [],
            "totalBenefit": 0.0,
            "totalDistanceKm": 0.0,
        }

        if optimal:
            selectedCircuits: List[str] = [
                circuit
                for circuit in self.circuits
                if pyo.value(model.x[circuit]) > 0.5
            ]

            results["selectedCircuits"] = selectedCircuits
            results["totalBenefit"] = sum(
                self.benefits[circuit] for circuit in selectedCircuits
            )

            calendar: List[Tuple[int, str, float]] = []
            for circuit in selectedCircuits:
                for week in range(1, self.nWeeks + 1):
                    if pyo.value(model.y[circuit, week]) > 0.5:
                        calendar.append((week, circuit, self.benefits[circuit]))
                        break

            calendar.sort(key=lambda item: item[0])
            results["calendar"] = calendar

            totalDistanceKm, travelLegs = self._calculateCalendarDistance(calendar)
            results["totalDistanceKm"] = totalDistanceKm
            results["travelLegs"] = travelLegs

        return results

    def printResults(self, results: Dict[str, Any]) -> None:
        print("\n" + "=" * 120)
        print("RESULTADOS DE LA OPTIMIZACIÓN")
        print("=" * 120)

        if results["objectiveValue"] is None:
            print("No se encontró solución óptima")
            return

        print("\nESTADO: Solución Óptima Encontrada")
        print(f"BENEFICIO TOTAL: {results['totalBenefit']:.2f} (M USD)")
        print(f"NÚMERO DE CARRERAS: {len(results['selectedCircuits'])}")
        print(f"DISTANCIA TOTAL DEL CALENDARIO: {results['totalDistanceKm']:.2f} km")

        print("\n" + "-" * 120)
        print(f"{'Semana':<8} {'Circuito':<40} {'Beneficio (M USD)':<20}")
        print("-" * 120)

        for week, circuit, benefit in results["calendar"]:
            circuitName = self.df[self.df["circuitId"] == circuit]["nombre"].values[0]
            print(f"{week:<8} {circuitName:<40} {benefit:<20.2f}")

        print("-" * 120)
        print(f"{'TOTAL':<8} {'':<40} {results['totalBenefit']:<20.2f}")
        print("=" * 120)

        if len(results["travelLegs"]) > 0:
            print("\nDETALLE DE DESPLAZAMIENTOS ENTRE CARRERAS")
            print("-" * 140)
            print(
                f"{'Semana origen':<14} {'Circuito origen':<30} "
                f"{'Semana destino':<14} {'Circuito destino':<30} {'Distancia (km)':<18}"
            )
            print("-" * 140)

            for week1, circuit1, week2, circuit2, distanceKm in results["travelLegs"]:
                circuitName1 = self.df[self.df["circuitId"] == circuit1]["nombre"].values[0]
                circuitName2 = self.df[self.df["circuitId"] == circuit2]["nombre"].values[0]

                print(
                    f"{week1:<14} {circuitName1:<30} "
                    f"{week2:<14} {circuitName2:<30} {distanceKm:<18.2f}"
                )

            print("-" * 140)
            print(f"{'DISTANCIA TOTAL':<90} {results['totalDistanceKm']:.2f} km")
            print("=" * 140)

    def getResultsDataFrame(self, results: Dict[str, Any]) -> pd.DataFrame:
        rows = []
        for week, circuit, benefit in results["calendar"]:
            circuitName = self.df[self.df["circuitId"] == circuit]["nombre"].values[0]
            rows.append({
                "Semana": week,
                "CircuitId": circuit,
                "Circuito": circuitName,
                "Beneficio_M_USD": benefit
            })
        return pd.DataFrame(rows)


def main() -> None:
    optimizer = F1CalendarOptimizer(
        "circuitos_f1_beneficios_disponible_simulados.csv"
    )

    # Si quieres fijar primero y último circuito:
    results = optimizer.optimize(firstCircuit="avus", lastCircuit="jarama")

    #results: Dict[str, Any] = optimizer.optimize()
    optimizer.printResults(results)


if __name__ == "__main__":
    main()