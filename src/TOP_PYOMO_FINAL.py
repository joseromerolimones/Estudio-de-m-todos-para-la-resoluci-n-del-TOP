"""
Versión 3: Maximiza beneficio y minimiza la distancia total recorrida
entre carreras consecutivas del calendario seleccionado.
"""

import math
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition


class F1CalendarOptimizer:
    def __init__(self, csvPath: str, alfa: float = 0.6) -> None:
        self.df: pd.DataFrame = pd.read_csv(csvPath)
        self.circuits: List[str] = self.df["circuitId"].tolist()
        self.nCircuits: int = len(self.circuits)
        self.nWeeks: int = 52
        self.nRaces: int = 24
        self.alfa: float = alfa

        self.benefits: Dict[str, float] = dict(
            zip(self.df["circuitId"], self.df["beneficio_total_millones_usd"])
        )

        self.distances: Dict[Tuple[str, str], float] = self._calculateDistanceMatrix()
        self.unavailableWeeks: Dict[str, Set[int]] = self._parseUnavailableWeeks()

        self.model: pyo.ConcreteModel | None = None
        self._solverResults: Any = None

    def _calculateHaversineDistance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        r: float = 6371.0
        lat1Rad: float = math.radians(lat1)
        lon1Rad: float = math.radians(lon1)
        lat2Rad: float = math.radians(lat2)
        lon2Rad: float = math.radians(lon2)
        dLat: float = lat2Rad - lat1Rad
        dLon: float = lon2Rad - lon1Rad

        a: float = (
            math.sin(dLat / 2) ** 2
            + math.cos(lat1Rad) * math.cos(lat2Rad) * math.sin(dLon / 2) ** 2
        )
        c: float = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r * c

    def _calculateDistanceMatrix(self) -> Dict[Tuple[str, str], float]:
        distances: Dict[Tuple[str, str], float] = {}

        for i, circuitI in enumerate(self.circuits):
            latI: float = self.df.iloc[i]["latitud"]
            lonI: float = self.df.iloc[i]["longitud"]

            for j, circuitJ in enumerate(self.circuits):
                if i != j:
                    latJ: float = self.df.iloc[j]["latitud"]
                    lonJ: float = self.df.iloc[j]["longitud"]
                    distance: float = self._calculateHaversineDistance(
                        latI, lonI, latJ, lonJ
                    )
                    distances[(circuitI, circuitJ)] = distance

        return distances

    def _parseUnavailableWeeks(self) -> Dict[str, Set[int]]:
        unavailable: Dict[str, Set[int]] = {}

        for _, row in self.df.iterrows():
            circuitId: str = row["circuitId"]
            weeksStr = row["semanas_no_disponibles"]

            weeks: Set[int] = set()
            if isinstance(weeksStr, str):
                weeks = {int(week.strip()) for week in weeksStr.split(",") if week.strip()}

            unavailable[circuitId] = weeks

        return unavailable

    def _createModel(self) -> None:
        model = pyo.ConcreteModel()

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

        model.ARC_INDEX = pyo.Set(
            dimen=4,
            initialize=[
                (i, j, t, u)
                for i in self.circuits
                for j in self.circuits
                if i != j
                for t in range(1, self.nWeeks)
                for u in range(t + 1, self.nWeeks + 1)
            ],
        )

        model.benefit = pyo.Param(
            model.CIRCUITS,
            initialize=self.benefits,
            within=pyo.Reals,
        )

        model.distance = pyo.Param(
            model.CIRCUITS,
            model.CIRCUITS,
            initialize=lambda m, i, j: 0.0 if i == j else self.distances[(i, j)],
            within=pyo.NonNegativeReals,
            default=0.0,
        )

        model.x = pyo.Var(model.CIRCUITS, domain=pyo.Binary)
        model.y = pyo.Var(model.CIRCUITS, model.WEEKS, domain=pyo.Binary)

        # z[i,j,t,u] = 1 si i está en t, j está en u y j es la siguiente carrera tras i
        model.z = pyo.Var(model.ARC_INDEX, domain=pyo.Binary)

        self.model = model

    def _addConstraints(
        self,
        firstCircuit: str | None = None,
        lastCircuit: str | None = None,
    ) -> None:
        if self.model is None:
            raise RuntimeError("Modelo no creado")

        model = self.model

        def totalRacesRule(m):
            return sum(m.x[circuit] for circuit in m.CIRCUITS) <= self.nRaces

        model.totalRaces = pyo.Constraint(rule=totalRacesRule)

        def assignmentRule(m, circuit):
            return sum(m.y[circuit, week] for week in m.WEEKS) == m.x[circuit]

        model.assignment = pyo.Constraint(model.CIRCUITS, rule=assignmentRule)

        def unavailableRule(m, circuit, week):
            return m.y[circuit, week] == 0

        if len(list(model.UNAVAILABLE)) > 0:
            model.unavailable = pyo.Constraint(
                model.UNAVAILABLE,
                rule=unavailableRule,
            )

        def oneRaceWeekRule(m, week):
            return sum(m.y[circuit, week] for circuit in m.CIRCUITS) <= 1

        model.oneRaceWeek = pyo.Constraint(model.WEEKS, rule=oneRaceWeekRule)

        def max3RacesIn4WeeksRule(m, startWeek):
            return sum(
                m.y[circuit, startWeek + k]
                for circuit in m.CIRCUITS
                for k in range(4)
            ) <= 3

        model.max3RacesIn4Weeks = pyo.Constraint(
            model.WINDOW_START,
            rule=max3RacesIn4WeeksRule,
        )

        if firstCircuit is not None and firstCircuit in self.circuits:
            model.firstCircuitSelected = pyo.Constraint(
                expr=model.x[firstCircuit] == 1
            )
            model.firstCircuitWeek1 = pyo.Constraint(
                expr=model.y[firstCircuit, 1] == 1
            )

        if lastCircuit is not None and lastCircuit in self.circuits:
            model.lastCircuitSelected = pyo.Constraint(
                expr=model.x[lastCircuit] == 1
            )
            model.lastCircuitWeek52 = pyo.Constraint(
                expr=model.y[lastCircuit, self.nWeeks] == 1
            )

        # z solo puede activarse si i está en t y j está en u
        def zUpperIRule(m, i, j, t, u):
            return m.z[i, j, t, u] <= m.y[i, t]

        def zUpperJRule(m, i, j, t, u):
            return m.z[i, j, t, u] <= m.y[j, u]

        model.zUpperI = pyo.Constraint(model.ARC_INDEX, rule=zUpperIRule)
        model.zUpperJ = pyo.Constraint(model.ARC_INDEX, rule=zUpperJRule)

        # Si z=1, no puede haber carreras entre t y u
        def zNoIntermediateRule(m, i, j, t, u):
            if u == t + 1:
                return pyo.Constraint.Skip
            return m.z[i, j, t, u] <= 1 - sum(
                m.y[circuit, week]
                for circuit in m.CIRCUITS
                for week in range(t + 1, u)
            )

        model.zNoIntermediate = pyo.Constraint(
            model.ARC_INDEX,
            rule=zNoIntermediateRule,
        )

        # Si hay carrera en t y existe alguna carrera posterior, debe haber exactamente un sucesor
        def uniqueSuccessorRule(m, i, t):
            laterArcs = [
                (ii, jj, tt, uu)
                for (ii, jj, tt, uu) in m.ARC_INDEX
                if ii == i and tt == t
            ]
            if not laterArcs:
                return pyo.Constraint.Skip
            return sum(m.z[ii, jj, tt, uu] for (ii, jj, tt, uu) in laterArcs) <= m.y[i, t]

        model.uniqueSuccessor = pyo.Constraint(
            model.CIRCUITS,
            model.WEEKS,
            rule=uniqueSuccessorRule,
        )

        # Si hay carrera en u y existe alguna carrera anterior, debe haber exactamente un predecesor
        def uniquePredecessorRule(m, j, u):
            previousArcs = [
                (ii, jj, tt, uu)
                for (ii, jj, tt, uu) in m.ARC_INDEX
                if jj == j and uu == u
            ]
            if not previousArcs:
                return pyo.Constraint.Skip
            return sum(m.z[ii, jj, tt, uu] for (ii, jj, tt, uu) in previousArcs) <= m.y[j, u]

        model.uniquePredecessor = pyo.Constraint(
            model.CIRCUITS,
            model.WEEKS,
            rule=uniquePredecessorRule,
        )

        # Activa exactamente un arco entre cada pareja de carreras consecutivas reales del calendario
        def totalArcsRule(m):
            return sum(m.z[index] for index in m.ARC_INDEX) == sum(
                m.x[circuit] for circuit in m.CIRCUITS
            ) - 1

        model.totalArcs = pyo.Constraint(rule=totalArcsRule)

    def _setObjective(self) -> None:
        if self.model is None:
            raise RuntimeError("Modelo no creado")

        model = self.model

        benefitComponent = sum(
            model.benefit[circuit] * model.x[circuit]
            for circuit in model.CIRCUITS
        )

        distanceComponent = sum(
            model.distance[i, j] * model.z[i, j, t, u]
            for (i, j, t, u) in model.ARC_INDEX
        )

        model.objective = pyo.Objective(
            expr=(
                self.alfa *  distanceComponent
                + (1 - self.alfa) * benefitComponent
            ),
            sense=pyo.maximize,
        )

    def optimize(
        self,
        firstCircuit: str | None = None,
        lastCircuit: str | None = None,
    ) -> Dict[str, Any]:
        print("=" * 80)
        print("F1 GRAND PRIX CALENDAR OPTIMIZATION (v3)")
        print("=" * 80)

        print("\n[1/5] Creando modelo de optimización...")
        self._createModel()

        print("[2/5] Añadiendo restricciones...")
        self._addConstraints(firstCircuit, lastCircuit)

        print("[3/5] Estableciendo función objetivo...")
        print(f" Parámetro alfa = {self.alfa}")
        print(
            f" Énfasis: {self.alfa * 100:.0f}% Beneficio - {(1 - self.alfa) * 100:.0f}% Distancia"
        )
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
            "totalBenefit": 0.0,
            "totalDistance": 0.0,
            "alfa": self.alfa,
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
            totalDistance: float = 0.0

            for circuit in selectedCircuits:
                for week in range(1, self.nWeeks + 1):
                    if pyo.value(model.y[circuit, week]) > 0.5:
                        calendar.append((week, circuit, self.benefits[circuit]))
                        break

            calendar.sort(key=lambda item: item[0])

            for idx in range(len(calendar) - 1):
                _, circuitI, _ = calendar[idx]
                _, circuitJ, _ = calendar[idx + 1]
                totalDistance += self.distances.get((circuitI, circuitJ), 0.0)

            results["calendar"] = calendar
            results["totalDistance"] = totalDistance

        return results

    def printResults(self, results: Dict[str, Any]) -> None:
        print("\n" + "=" * 100)
        print("RESULTADOS DE LA OPTIMIZACIÓN")
        print("=" * 100)

        if results["objectiveValue"] is None:
            print("No se encontró solución óptima")
            return

        print("\nESTADO: Solución Óptima Encontrada")
        print(f"VALOR FUNCIÓN OBJETIVO: {results['objectiveValue']:.2f}")
        print(f"BENEFICIO TOTAL: {results['totalBenefit']:.2f} (M USD)")
        print(f"DISTANCIA TOTAL: {results['totalDistance']:.2f} (km)")
        print(f"NÚMERO DE CARRERAS: {len(results['selectedCircuits'])}")

        print("\n" + "-" * 120)
        print(
            f"{'Semana':<8} {'Circuito':<35} {'País':<20} {'Beneficio (M USD)':<20} {'Distancia (km)':<20}"
        )
        print("-" * 120)

        for idx, (week, circuit, benefit) in enumerate(results["calendar"]):
            circuitName: str = self.df[self.df["circuitId"] == circuit]["nombre"].values[0]
            country: str = self.df[self.df["circuitId"] == circuit]["pais"].values[0]

            if idx < len(results["calendar"]) - 1:
                nextCircuit: str = results["calendar"][idx + 1][1]
                distToNext: float = self.distances.get((circuit, nextCircuit), 0.0)
                distStr: str = f"{distToNext:.2f}"
            else:
                distStr = "-"

            print(
                f"{week:<8} {circuitName:<35} {country:<20} {benefit:>15.2f} {distStr:>15}"
            )

        print("-" * 120)
        print(
            f"{'TOTAL':<8} {'':<35} {'':<20} {results['totalBenefit']:>15.2f} {results['totalDistance']:>15.2f}"
        )
        print("=" * 100)


def main() -> None:
    optimizer = F1CalendarOptimizer(
        "circuitos_f1_beneficios_disponible_simulados.csv",
        alfa=0.5,
    )

    results: Dict[str, Any] = optimizer.optimize()
    optimizer.printResults(results)


if __name__ == "__main__":
    main()