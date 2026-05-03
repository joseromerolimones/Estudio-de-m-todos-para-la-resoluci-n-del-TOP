import csv
import heapq
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ============================================================
# PARÁMETROS
# ============================================================
R: float = 6371.0
NOMBRE_CIRCUITO_ORIGEN: str = "AVUS"
NOMBRE_CIRCUITO_DESTINO: str = "Jarama"

NUM_SEMANAS: int = 52
MAX_FINES_SEGUIDOS: int = 3
NUM_CARRERAS: int = 24
NUM_VEHICULOS: int = 1
T_MAX: float = 3000000.0
VEL: float = 1.0

RUTA_CSV_CIRCUITOS: str = "circuitos_f1_beneficios_disponible_simulados.csv"

# VNS determinista
ALPHAS: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BETA: float = 0.30
K_MAX: int = 3
MAX_TIME_SEC: int = 30
T0_SA: float = 1000.0
LAMBDA_SA: float = 0.999
RANDOM_SEED: int = 42

# LS1 nuevo: búsqueda local compatible con calendario fijo
LS1_MAX_CANDIDATOS: int = 150

# Debug opcional
DEBUG_VNS: bool = False
DEBUG_LS1: bool = False

# ============================================================
# MODELO DE DATOS
# ============================================================
class Nodo:
    __slots__ = [
        "id",
        "nombre",
        "ciudad",
        "pais",
        "x",
        "y",
        "recompensa",
        "mercado",
        "tipo",
        "beneficioDict",
        "tipoNodo",
        "semanasProhibidas",
        "semanaAsignada",
        "circuitoBaseNombre",
        "latRad",
        "lonRad",
        "sinLat",
        "cosLat",
    ]

    def __init__(
        self,
        id: int,
        nombre: str,
        ciudad: str,
        pais: str,
        x: float,
        y: float,
        recompensa: int,
        mercado: str,
        tipo: str,
        beneficioDict: Dict[str, float],
        tipoNodo: str = "cliente",
        semanasProhibidas: Optional[List[int]] = None,
    ) -> None:
        self.id = id
        self.nombre = nombre
        self.ciudad = ciudad
        self.pais = pais
        self.x = x
        self.y = y
        self.recompensa = recompensa
        self.mercado = mercado
        self.tipo = tipo
        self.beneficioDict = beneficioDict
        self.tipoNodo = tipoNodo
        self.semanasProhibidas = semanasProhibidas if semanasProhibidas else []
        self.semanaAsignada: Optional[int] = None
        self.circuitoBaseNombre: Optional[str] = None

        self.latRad = math.radians(y)
        self.lonRad = math.radians(x)
        self.sinLat = math.sin(self.latRad)
        self.cosLat = math.cos(self.latRad)

    def __repr__(self) -> str:
        return (
            f"Nodo({self.id}, {self.nombre}, "
            f"semana={self.semanaAsignada}, beneficio={self.recompensa})"
        )


@dataclass
class EstadoCalendario:
    semanasSorted: List[int] = field(default_factory=list)
    semanasSet: Set[int] = field(default_factory=set)
    semanaToNodo: Dict[int, int] = field(default_factory=dict)

    rachaInicio: int = 0
    rachaFin: int = 0
    rachaMax: int = 0

    beneficioReal: int = 0
    numCarreras: int = 0

    @classmethod
    def vacio(cls) -> "EstadoCalendario":
        return cls()

    @classmethod
    def desdeNodo(
        cls, semana: int, nodoId: int, beneficio: int
    ) -> "EstadoCalendario":
        return cls(
            semanasSorted=[semana],
            semanasSet={semana},
            semanaToNodo={semana: nodoId},
            rachaInicio=1,
            rachaFin=1,
            rachaMax=1,
            beneficioReal=beneficio,
            numCarreras=1,
        )

    def copiar(self) -> "EstadoCalendario":
        return EstadoCalendario(
            semanasSorted=self.semanasSorted.copy(),
            semanasSet=self.semanasSet.copy(),
            semanaToNodo=self.semanaToNodo.copy(),
            rachaInicio=self.rachaInicio,
            rachaFin=self.rachaFin,
            rachaMax=self.rachaMax,
            beneficioReal=self.beneficioReal,
            numCarreras=self.numCarreras,
        )


class Ruta:
    __slots__ = [
        "id",
        "nodos",
        "tiempoTotal",
        "recompensaTotal",
        "nodosVisitados",
        "circuitosBaseVisitados",
        "estadoCal",
        "tieneMandatorias",
    ]

    def __init__(self, idRuta: int) -> None:
        self.id = idRuta
        self.nodos: List[int] = [0, 0]
        self.tiempoTotal: float = 0.0
        self.recompensaTotal: int = 0
        self.nodosVisitados: Set[int] = set()
        self.circuitosBaseVisitados: Set[str] = set()
        self.estadoCal: EstadoCalendario = EstadoCalendario.vacio()
        self.tieneMandatorias: bool = False

    def esValida(self, tMax: float) -> bool:
        return self.tiempoTotal <= tMax

    def clientes(self) -> List[int]:
        return self.nodos[1:-1]

    def beneficioReal(self) -> int:
        return self.estadoCal.beneficioReal

    def __repr__(self) -> str:
        return (
            f"Ruta {self.id}: nodos={len(self.nodos) - 2}, "
            f"T={self.tiempoTotal:.2f}, Rbruta={self.recompensaTotal}, "
            f"Carreras={self.estadoCal.numCarreras}, "
            f"BenefReal={self.estadoCal.beneficioReal}, "
            f"Mandatorias={self.tieneMandatorias}"
        )


@dataclass
class DatosProblema:
    nodos: List[Nodo]
    tiempo: List[List[float]]
    origenId: int
    destinoId: int
    mandInicioId: int
    mandFinId: int
    semanaMin: int
    semanaMax: int
    elegiblesIds: List[int]
    customerIds: List[int]
    maxFines: int
    maxCarreras: int
    tMax: float


# ============================================================
# DISTANCIAS
# ============================================================
class GestorDistancias:
    def __init__(self, nodosBase: List[Nodo]) -> None:
        self.cacheDistancias: Dict[Tuple[str, str], float] = {}
        self._precalcular(nodosBase)

    def _precalcular(self, nodosBase: List[Nodo]) -> None:
        n = len(nodosBase)
        for i in range(n):
            for j in range(i, n):
                nodo1, nodo2 = nodosBase[i], nodosBase[j]
                distancia = 0.0 if i == j else self._haversine(nodo1, nodo2)
                self.cacheDistancias[(nodo1.nombre, nodo2.nombre)] = distancia
                self.cacheDistancias[(nodo2.nombre, nodo1.nombre)] = distancia

    def _haversine(self, nodo1: Nodo, nodo2: Nodo) -> float:
        dLat = nodo2.latRad - nodo1.latRad
        dLon = nodo2.lonRad - nodo1.lonRad
        a = (
            math.sin(dLat / 2) ** 2
            + nodo1.cosLat * nodo2.cosLat * math.sin(dLon / 2) ** 2
        )
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def obtenerDistancia(self, nombre1: str, nombre2: str) -> float:
        return self.cacheDistancias.get((nombre1, nombre2), 0.0)


# ============================================================
# LECTURA / PREPROCESADO
# ============================================================
def leerCircuitosCsv(rutaArchivo: str) -> List[Nodo]:
    nodosBase: List[Nodo] = []

    try:
        with open(rutaArchivo, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for idx, fila in enumerate(reader, start=1):
                beneficioDict = {
                    "Merchandising": float(fila["merchandising_millones_usd"]),
                    "Beneficio Imagen": float(
                        fila["beneficio_imagen_millones_usd"]
                    ),
                    "Patrocinio Local": float(
                        fila["patrocinio_local_millones_usd"]
                    ),
                    "Hospitalidad": float(fila["hospitalidad_millones_usd"]),
                    "Ticketing": float(fila["ticketing_millones_usd"]),
                }

                raw = fila.get("semanas_no_disponibles", "").strip()
                semanasProhibidas: List[int] = []
                if raw:
                    semanasProhibidas = [
                        int(s.strip()) for s in raw.split(",") if s.strip().isdigit()
                    ]

                nodo = Nodo(
                    id=idx,
                    nombre=fila["nombre"].strip(),
                    ciudad=fila["ciudad"].strip(),
                    pais=fila["pais"].strip(),
                    x=float(fila["longitud"]),
                    y=float(fila["latitud"]),
                    recompensa=int(
                        round(float(fila["beneficio_total_millones_usd"]))
                    ),
                    mercado=fila["mercado"].strip(),
                    tipo=fila["tipo"].strip(),
                    beneficioDict=beneficioDict,
                    tipoNodo="cliente",
                    semanasProhibidas=semanasProhibidas,
                )
                nodo.circuitoBaseNombre = nodo.nombre
                nodosBase.append(nodo)

    except FileNotFoundError:
        print(f"Error: no se encontró el archivo {rutaArchivo}")
        return []

    print(f"Se cargaron {len(nodosBase)} circuitos base del CSV")
    return nodosBase


def expandirNodosPorSemana(
    nodosBase: List[Nodo], numSemanas: int = 52
) -> List[Nodo]:
    nodosExpandidos: List[Nodo] = []

    for nodoBase in nodosBase:
        semanasDisponibles = sorted(
            set(range(1, numSemanas + 1)) - set(nodoBase.semanasProhibidas)
        )
        for semana in semanasDisponibles:
            nodoExpandido = Nodo(
                id=0,
                nombre=f"{nodoBase.nombre}_semana_{semana}",
                ciudad=nodoBase.ciudad,
                pais=nodoBase.pais,
                x=nodoBase.x,
                y=nodoBase.y,
                recompensa=nodoBase.recompensa,
                mercado=nodoBase.mercado,
                tipo=nodoBase.tipo,
                beneficioDict=nodoBase.beneficioDict.copy(),
                tipoNodo="cliente",
                semanasProhibidas=[
                    s for s in range(1, numSemanas + 1) if s != semana
                ],
            )
            nodoExpandido.semanaAsignada = semana
            nodoExpandido.circuitoBaseNombre = nodoBase.nombre
            nodosExpandidos.append(nodoExpandido)

    print("\n" + "=" * 70)
    print("EXPANSIÓN DE NODOS POR SEMANA")
    print("=" * 70)
    print(f"Nodos base: {len(nodosBase)}")
    print(f"Nodos expandidos: {len(nodosExpandidos)}")
    print("=" * 70 + "\n")
    return nodosExpandidos


def obtenerCoordCircuito(
    nodosBase: List[Nodo], nombre: str
) -> Tuple[float, float]:
    for nodo in nodosBase:
        if nodo.nombre == nombre:
            return nodo.x, nodo.y
    raise ValueError(f"No se encontró un circuito con nombre '{nombre}'.")


def agregarNodosEspeciales(
    nodosClientes: List[Nodo],
    origenCoord: Tuple[float, float],
    destinoCoord: Tuple[float, float],
) -> List[Nodo]:
    nodosLocal: List[Nodo] = []

    nodoOrigen = Nodo(
        id=0,
        nombre="ORIGEN - Depósito",
        ciudad="Depósito",
        pais="N/A",
        x=origenCoord[0],
        y=origenCoord[1],
        recompensa=0,
        mercado="N/A",
        tipo="Especial",
        beneficioDict={},
        tipoNodo="origen",
    )
    nodoOrigen.circuitoBaseNombre = NOMBRE_CIRCUITO_ORIGEN
    nodosLocal.append(nodoOrigen)

    for idx, nodoCliente in enumerate(nodosClientes, start=1):
        nodoCliente.id = idx
        nodosLocal.append(nodoCliente)

    nodoDestino = Nodo(
        id=len(nodosLocal),
        nombre="DESTINO - Depósito",
        ciudad="Depósito",
        pais="N/A",
        x=destinoCoord[0],
        y=destinoCoord[1],
        recompensa=0,
        mercado="N/A",
        tipo="Especial",
        beneficioDict={},
        tipoNodo="destino",
    )
    nodoDestino.circuitoBaseNombre = NOMBRE_CIRCUITO_DESTINO
    nodosLocal.append(nodoDestino)

    return nodosLocal


def crearMatrizTiemposOptimizada(
    nodos: List[Nodo],
    gestor: GestorDistancias,
    velocidad: float,
) -> List[List[float]]:
    n = len(nodos)
    tiempo = [[0.0] * n for _ in range(n)]

    for i in range(n):
        nombreI = nodos[i].circuitoBaseNombre or nodos[i].nombre
        for j in range(i + 1, n):
            nombreJ = nodos[j].circuitoBaseNombre or nodos[j].nombre
            distancia = gestor.obtenerDistancia(nombreI, nombreJ)
            t = round(distancia / velocidad, 2)
            tiempo[i][j] = t
            tiempo[j][i] = t

    return tiempo


# ============================================================
# CALENDARIO
# ============================================================
def _calcularRachaStats(semanasSorted: List[int]) -> Tuple[int, int, int]:
    if not semanasSorted:
        return 0, 0, 0
    if len(semanasSorted) == 1:
        return 1, 1, 1

    rachaActual = 1
    rachaMax = 1
    rachaInicio = 1
    inicioLocked = False

    for i in range(1, len(semanasSorted)):
        if semanasSorted[i] == semanasSorted[i - 1] + 1:
            rachaActual += 1
            if rachaActual > rachaMax:
                rachaMax = rachaActual
            if not inicioLocked:
                rachaInicio = rachaActual
        else:
            inicioLocked = True
            rachaActual = 1

    rachaFin = rachaActual
    return rachaInicio, rachaFin, rachaMax


def fusionarEstadosCalendario(
    estadoA: EstadoCalendario,
    estadoB: EstadoCalendario,
    maxFines: int,
    maxCarreras: int,
) -> Optional[EstadoCalendario]:
    if not estadoA.semanasSorted:
        if estadoB.numCarreras <= maxCarreras and estadoB.rachaMax <= maxFines:
            return estadoB.copiar()
        return None

    if not estadoB.semanasSorted:
        if estadoA.numCarreras <= maxCarreras and estadoA.rachaMax <= maxFines:
            return estadoA.copiar()
        return None

    if not estadoA.semanasSet.isdisjoint(estadoB.semanasSet):
        return None

    if estadoA.numCarreras + estadoB.numCarreras > maxCarreras:
        return None

    if estadoA.semanasSorted and estadoB.semanasSorted:
        if estadoA.semanasSorted[-1] + 1 == estadoB.semanasSorted[0]:
            if estadoA.rachaFin + estadoB.rachaInicio > maxFines:
                return None
        if estadoB.semanasSorted[-1] + 1 == estadoA.semanasSorted[0]:
            if estadoB.rachaFin + estadoA.rachaInicio > maxFines:
                return None

    mergedSemanas = list(heapq.merge(estadoA.semanasSorted, estadoB.semanasSorted))
    rachaInicio, rachaFin, rachaMax = _calcularRachaStats(mergedSemanas)
    if rachaMax > maxFines:
        return None

    semanaToNodo = estadoA.semanaToNodo.copy()
    semanaToNodo.update(estadoB.semanaToNodo)

    return EstadoCalendario(
        semanasSorted=mergedSemanas,
        semanasSet=set(mergedSemanas),
        semanaToNodo=semanaToNodo,
        rachaInicio=rachaInicio,
        rachaFin=rachaFin,
        rachaMax=rachaMax,
        beneficioReal=estadoA.beneficioReal + estadoB.beneficioReal,
        numCarreras=estadoA.numCarreras + estadoB.numCarreras,
    )


# ============================================================
# UTILIDADES DE RUTA
# ============================================================
def beneficioRuta(ruta: Ruta) -> int:
    return ruta.estadoCal.beneficioReal


def esMejorRuta(candidata: Optional[Ruta], incumbente: Optional[Ruta]) -> bool:
    if candidata is None:
        return False
    if incumbente is None:
        return True

    beneficioCandidata = beneficioRuta(candidata)
    beneficioIncumbente = beneficioRuta(incumbente)

    if beneficioCandidata != beneficioIncumbente:
        return beneficioCandidata > beneficioIncumbente
    if abs(candidata.tiempoTotal - incumbente.tiempoTotal) > 1e-9:
        return candidata.tiempoTotal < incumbente.tiempoTotal
    return len(candidata.clientes()) > len(incumbente.clientes())


def buildRouteFromClientIds(
    clientIds: List[int],
    datos: DatosProblema,
    routeId: int = 0,
) -> Optional[Ruta]:
    ids = sorted(
        set(clientIds),
        key=lambda nodoId: (
            datos.nodos[nodoId].semanaAsignada
            if datos.nodos[nodoId].semanaAsignada is not None
            else 10**9
        ),
    )

    if len(ids) > datos.maxCarreras:
        return None

    ruta = Ruta(routeId)
    ruta.nodos = [datos.origenId] + ids + [datos.destinoId]
    ruta.nodosVisitados = set(ruta.nodos)
    ruta.recompensaTotal = 0
    ruta.tiempoTotal = 0.0
    ruta.estadoCal = EstadoCalendario.vacio()
    ruta.circuitosBaseVisitados = set()

    for nodoId in ids:
        nodo = datos.nodos[nodoId]
        semana = nodo.semanaAsignada
        base = nodo.circuitoBaseNombre or nodo.nombre

        if semana is None:
            return None

        if base in ruta.circuitosBaseVisitados:
            return None

        nuevoEstadoCal = fusionarEstadosCalendario(
            ruta.estadoCal,
            EstadoCalendario.desdeNodo(semana, nodoId, nodo.recompensa),
            datos.maxFines,
            datos.maxCarreras,
        )
        if nuevoEstadoCal is None:
            return None

        ruta.estadoCal = nuevoEstadoCal
        ruta.circuitosBaseVisitados.add(base)
        ruta.recompensaTotal += nodo.recompensa

    ruta.tiempoTotal = sum(
        datos.tiempo[ruta.nodos[k]][ruta.nodos[k + 1]]
        for k in range(len(ruta.nodos) - 1)
    )
    if ruta.tiempoTotal > datos.tMax:
        return None

    clientesSet = set(ids)
    ruta.tieneMandatorias = (
        datos.mandInicioId in clientesSet and datos.mandFinId in clientesSet
    )
    return ruta


def cloneRoute(ruta: Ruta, datos: DatosProblema, routeId: int = 0) -> Ruta:
    nuevaRuta = buildRouteFromClientIds(ruta.clientes(), datos, routeId)
    if nuevaRuta is None:
        raise ValueError("No se pudo clonar la ruta")
    return nuevaRuta


def clientesRemovibles(ruta: Ruta, datos: DatosProblema) -> List[int]:
    return [
        nodoId
        for nodoId in ruta.clientes()
        if nodoId not in {datos.mandInicioId, datos.mandFinId}
    ]


def costeMarginalNodo(ruta: Ruta, nodoId: int, datos: DatosProblema) -> float:
    clientes = ruta.clientes()
    pos = clientes.index(nodoId)
    prevId = datos.origenId if pos == 0 else clientes[pos - 1]
    nextId = datos.destinoId if pos == len(clientes) - 1 else clientes[pos + 1]

    return (
        datos.tiempo[prevId][nodoId]
        + datos.tiempo[nodoId][nextId]
        - datos.tiempo[prevId][nextId]
    )


def ordenarRemoviblesParaExchange(
    ruta: Ruta, datos: DatosProblema
) -> List[int]:
    removibles = clientesRemovibles(ruta, datos)
    removibles.sort(
        key=lambda nodoId: (
            datos.nodos[nodoId].recompensa,
            -costeMarginalNodo(ruta, nodoId, datos),
        )
    )
    return removibles


def generarCandidatosExchangeCalendario(
    ruta: Ruta,
    outId: int,
    datos: DatosProblema,
    maxCandidatos: int = LS1_MAX_CANDIDATOS,
) -> List[int]:
    semanaOut = datos.nodos[outId].semanaAsignada
    baseOut = datos.nodos[outId].circuitoBaseNombre or datos.nodos[outId].nombre

    if semanaOut is None:
        return []

    basesOcupadas = set(ruta.circuitosBaseVisitados)
    basesOcupadas.discard(baseOut)

    candidatos: List[Tuple[Tuple[int, int, int, float], int]] = []

    for nodoId in datos.elegiblesIds:
        if nodoId in ruta.nodosVisitados:
            continue

        nodo = datos.nodos[nodoId]
        semanaIn = nodo.semanaAsignada
        baseIn = nodo.circuitoBaseNombre or nodo.nombre

        if semanaIn is None:
            continue
        if baseIn in basesOcupadas:
            continue

        gapSemana = abs(semanaIn - semanaOut)
        mismaSemana = 0 if semanaIn == semanaOut else 1
        gainRecompensa = nodo.recompensa - datos.nodos[outId].recompensa
        proxyDist = datos.tiempo[outId][nodoId]

        clave = (
            mismaSemana,
            gapSemana,
            -gainRecompensa,
            proxyDist,
        )
        candidatos.append((clave, nodoId))

    candidatos.sort(key=lambda item: item[0])
    return [nodoId for _, nodoId in candidatos[:maxCandidatos]]


# ============================================================
# HEURÍSTICA DE AHORROS
# ============================================================
def calcularSavings(
    i: int,
    j: int,
    alpha: float,
    datos: DatosProblema,
) -> float:
    reward = datos.nodos[i].recompensa + datos.nodos[j].recompensa
    timeSaved = (
        datos.tiempo[i][datos.destinoId]
        + datos.tiempo[datos.origenId][j]
        - datos.tiempo[i][j]
    )
    return alpha * (timeSaved) + (1.0 - alpha) * reward


def construirSolucionInicialAhorros(
    datos: DatosProblema, alpha: float
) -> Ruta:
    seed = buildRouteFromClientIds(
        [datos.mandInicioId, datos.mandFinId],
        datos,
        routeId=0,
    )
    if seed is None:
        raise SystemExit(
            "Inviable: las carreras obligatorias no generan una seed factible"
        )

    rutas: List[Ruta] = [seed]
    nextRouteId = 1

    for clienteId in datos.elegiblesIds:
        singleton = buildRouteFromClientIds([clienteId], datos, routeId=nextRouteId)
        if singleton is not None:
            rutas.append(singleton)
            nextRouteId += 1

    nodoARuta: Dict[int, Ruta] = {}
    for ruta in rutas:
        for nodoId in ruta.clientes():
            nodoARuta[nodoId] = ruta

    savingsList: List[Tuple[float, int, int]] = []
    for idxI in range(len(datos.customerIds)):
        i = datos.customerIds[idxI]
        for idxJ in range(idxI + 1, len(datos.customerIds)):
            j = datos.customerIds[idxJ]
            savingsList.append((calcularSavings(i, j, alpha, datos), i, j))
            savingsList.append((calcularSavings(j, i, alpha, datos), j, i))

    savingsList.sort(reverse=True, key=lambda item: item[0])

    for _, i, j in savingsList:
        rutaI = nodoARuta.get(i)
        rutaJ = nodoARuta.get(j)

        if rutaI is None or rutaJ is None or rutaI.id == rutaJ.id:
            continue

        if not (rutaI.tieneMandatorias or rutaJ.tieneMandatorias):
            continue

        merged = buildRouteFromClientIds(
            rutaI.clientes() + rutaJ.clientes(),
            datos,
            routeId=nextRouteId,
        )
        if merged is None:
            continue

        if rutaI in rutas:
            rutas.remove(rutaI)
        if rutaJ in rutas:
            rutas.remove(rutaJ)
        rutas.append(merged)
        nextRouteId += 1

        for nodoId in rutaI.clientes():
            nodoARuta.pop(nodoId, None)
        for nodoId in rutaJ.clientes():
            nodoARuta.pop(nodoId, None)
        for nodoId in merged.clientes():
            nodoARuta[nodoId] = merged

    rutasMandatorias = [ruta for ruta in rutas if ruta.tieneMandatorias]
    if not rutasMandatorias:
        raise SystemExit("Error: no quedó ninguna ruta con las carreras obligatorias")

    rutasMandatorias.sort(
        key=lambda ruta: (beneficioRuta(ruta), -ruta.tiempoTotal),
        reverse=True,
    )
    return rutasMandatorias[0]


def generarSolucionInicialPorAlpha(
    datos: DatosProblema,
) -> Tuple[Ruta, float]:
    mejorRuta: Optional[Ruta] = None
    mejorAlpha: float = ALPHAS[0]

    for alpha in ALPHAS:
        ruta = construirSolucionInicialAhorros(datos, alpha)
        if esMejorRuta(ruta, mejorRuta):
            mejorRuta = ruta
            mejorAlpha = alpha

    if mejorRuta is None:
        raise SystemExit("No se pudo generar una solución inicial factible")

    return mejorRuta, mejorAlpha


# ============================================================
# OPERADORES VNS
# ============================================================
def geometricBiasedIndex(n: int, beta: float, rng: random.Random) -> int:
    if n <= 1:
        return 0

    u = rng.random()
    idx = int(math.log(1 - u) / math.log(1 - beta))
    return min(idx, n - 1)


def evaluarInsercion(
    ruta: Ruta,
    nodoId: int,
    datos: DatosProblema,
) -> Optional[Tuple[float, Ruta]]:
    if nodoId in ruta.nodosVisitados:
        return None

    if len(ruta.clientes()) >= datos.maxCarreras:
        return None

    candidata = buildRouteFromClientIds(
        ruta.clientes() + [nodoId],
        datos,
        routeId=ruta.id,
    )
    if candidata is None:
        return None

    pos = candidata.nodos.index(nodoId)
    prevId = candidata.nodos[pos - 1]
    nextId = candidata.nodos[pos + 1]
    deltaT = (
        datos.tiempo[prevId][nodoId]
        + datos.tiempo[nodoId][nextId]
        - datos.tiempo[prevId][nextId]
    )
    recompensa = max(1, datos.nodos[nodoId].recompensa)
    ratio = deltaT / recompensa
    return ratio, candidata


def biasedInsertion(
    ruta: Ruta,
    candidateIds: List[int],
    datos: DatosProblema,
    beta: float,
    rng: random.Random,
) -> Ruta:
    actual = cloneRoute(ruta, datos, routeId=ruta.id)

    if len(actual.clientes()) >= datos.maxCarreras:
        return actual

    disponibles = [
        nodoId for nodoId in candidateIds if nodoId not in actual.nodosVisitados
    ]

    while True:
        if len(actual.clientes()) >= datos.maxCarreras:
            break

        evaluaciones: List[Tuple[float, int, Ruta]] = []
        for nodoId in disponibles:
            evaluacion = evaluarInsercion(actual, nodoId, datos)
            if evaluacion is not None:
                ratio, nuevaRuta = evaluacion
                evaluaciones.append((ratio, nodoId, nuevaRuta))

        if not evaluaciones:
            break

        evaluaciones.sort(key=lambda item: item[0])
        idx = geometricBiasedIndex(len(evaluaciones), beta, rng)
        _, elegido, nuevaRuta = evaluaciones[idx]

        if DEBUG_VNS:
            print(
                f"[BIASED_INSERT] elegido nid={elegido} "
                f"ratio={evaluaciones[idx][0]:.4f}"
            )

        actual = nuevaRuta
        disponibles = [
            nodoId
            for nodoId in disponibles
            if nodoId != elegido and nodoId not in actual.nodosVisitados
        ]

    return actual


def shaking(
    baseSol: Ruta,
    k: int,
    datos: DatosProblema,
    beta: float,
    rng: random.Random,
) -> Ruta:
    removibles = clientesRemovibles(baseSol, datos)
    if not removibles:
        return cloneRoute(baseSol, datos, routeId=baseSol.id)

    frac = min(0.05 * k, 0.30)
    q = max(1, math.ceil(frac * len(removibles)))
    q = min(q, len(removibles))

    aBorrar = set(rng.sample(removibles, q))
    parciales = [nodoId for nodoId in baseSol.clientes() if nodoId not in aBorrar]

    parcial = buildRouteFromClientIds(parciales, datos, routeId=baseSol.id)
    if parcial is None:
        parcial = buildRouteFromClientIds(
            [datos.mandInicioId, datos.mandFinId],
            datos,
            routeId=baseSol.id,
        )
    if parcial is None:
        return cloneRoute(baseSol, datos, routeId=baseSol.id)

    candidatos = [
        nodoId for nodoId in datos.elegiblesIds if nodoId not in parcial.nodosVisitados
    ]
    return biasedInsertion(parcial, candidatos, datos, beta, rng)


# ============================================================
# LS1 NUEVO: EXCHANGE COMPATIBLE CON CALENDARIO FIJO
# ============================================================
def localSearch1CalendarExchange(ruta: Ruta, datos: DatosProblema) -> Ruta:
    actual = cloneRoute(ruta, datos, routeId=ruta.id)

    while True:
        mejor = actual
        mejoro = False

        removibles = ordenarRemoviblesParaExchange(actual, datos)

        for outId in removibles:
            baseClientes = [nodoId for nodoId in actual.clientes() if nodoId != outId]
            candidatos = generarCandidatosExchangeCalendario(
                actual,
                outId,
                datos,
                maxCandidatos=LS1_MAX_CANDIDATOS,
            )

            if DEBUG_LS1:
                semanaOut = datos.nodos[outId].semanaAsignada
                print(
                    f"[LS1] probar sacar {outId} semana={semanaOut} "
                    f"recompensa={datos.nodos[outId].recompensa} "
                    f"con {len(candidatos)} candidatos"
                )

            for inId in candidatos:
                candidata = buildRouteFromClientIds(
                    baseClientes + [inId],
                    datos,
                    routeId=actual.id,
                )

                if esMejorRuta(candidata, mejor):
                    if DEBUG_LS1 and candidata is not None:
                        print(
                            f"[LS1] mejora: OUT {outId} ({datos.nodos[outId].nombre}) "
                            f"-> IN {inId} ({datos.nodos[inId].nombre}) | "
                            f"benef {beneficioRuta(actual)} -> {beneficioRuta(candidata)} | "
                            f"T {actual.tiempoTotal:.2f} -> {candidata.tiempoTotal:.2f}"
                        )
                    mejor = candidata  # type: ignore[assignment]
                    mejoro = True
                    break

            if mejoro:
                break

        if not mejoro:
            break

        actual = mejor

    return actual


def localSearch2Remove(
    ruta: Ruta,
    datos: DatosProblema,
    rng: random.Random,
) -> Ruta:
    removibles = clientesRemovibles(ruta, datos)
    if not removibles:
        return cloneRoute(ruta, datos, routeId=ruta.id)

    qMin = max(1, math.ceil(0.05 * len(removibles)))
    qMax = max(qMin, math.ceil(0.10 * len(removibles)))
    q = min(len(removibles), rng.randint(qMin, qMax))

    modo = rng.choice(["random", "high", "low"])

    if modo == "random":
        eliminar = set(rng.sample(removibles, q))
    elif modo == "high":
        eliminar = set(
            sorted(
                removibles,
                key=lambda nodoId: datos.nodos[nodoId].recompensa,
                reverse=True,
            )[:q]
        )
    else:
        eliminar = set(
            sorted(removibles, key=lambda nodoId: datos.nodos[nodoId].recompensa)[:q]
        )

    nueva = buildRouteFromClientIds(
        [nodoId for nodoId in ruta.clientes() if nodoId not in eliminar],
        datos,
        routeId=ruta.id,
    )
    return nueva if nueva is not None else cloneRoute(ruta, datos, routeId=ruta.id)


def localSearch3BiasedInsertion(
    ruta: Ruta,
    datos: DatosProblema,
    beta: float,
    rng: random.Random,
) -> Ruta:
    candidatos = [
        nodoId for nodoId in datos.elegiblesIds if nodoId not in ruta.nodosVisitados
    ]
    return biasedInsertion(ruta, candidatos, datos, beta, rng)


def probAceptacion(
    nuevoBenef: int,
    baseBenef: int,
    temperatura: float,
) -> float:
    if nuevoBenef >= baseBenef:
        return 1.0
    if temperatura <= 1e-12:
        return 0.0
    return math.exp((nuevoBenef - baseBenef) / temperatura)


# ============================================================
# VNS DETERMINISTA
# ============================================================
def vnsDeterminista(
    datos: DatosProblema,
    seed: Optional[int] = None,
    hacer_diagnostico: bool = False,
) -> Tuple[Ruta, float]:
    rng = random.Random(RANDOM_SEED if seed is None else seed)

    initSol, alphaFijo = generarSolucionInicialPorAlpha(datos)
    baseSol = cloneRoute(initSol, datos, routeId=initSol.id)
    bestSol = cloneRoute(initSol, datos, routeId=initSol.id)

    if hacer_diagnostico:
        diagnostico_k(initSol, datos) # type: ignore

    temperatura = T0_SA
    start = time.time()
    iteraciones = 0

    while time.time() - start <= MAX_TIME_SEC:
        k = 1

        while k <= K_MAX and time.time() - start <= MAX_TIME_SEC:
            iteraciones += 1

            newSol = shaking(baseSol, k, datos, BETA, rng)
            newSol = localSearch1CalendarExchange(newSol, datos)
            newSol = localSearch2Remove(newSol, datos, rng)
            newSol = localSearch3BiasedInsertion(newSol, datos, BETA, rng)

            if esMejorRuta(newSol, baseSol):
                baseSol = newSol

                if esMejorRuta(newSol, bestSol):
                    bestSol = cloneRoute(newSol, datos, routeId=newSol.id)

                k = 1
            else:
                p = probAceptacion(
                    beneficioRuta(newSol),
                    beneficioRuta(baseSol),
                    temperatura,
                )

                if rng.random() <= p:
                    baseSol = newSol
                    k = 1
                else:
                    k += 1

            temperatura *= LAMBDA_SA

    print("\n" + "=" * 70)
    print("VNS DETERMINISTA FINALIZADO")
    print("=" * 70)
    print(f"Iteraciones ejecutadas: {iteraciones}")
    print(f"Alpha seleccionado en la fase inicial: {alphaFijo}")
    print(f"Beneficio solución inicial: {beneficioRuta(initSol)}")
    print(f"Beneficio mejor solución VNS: {beneficioRuta(bestSol)}")
    print("=" * 70)

    return bestSol, alphaFijo


# ============================================================
# CALENDARIO FINAL
# ============================================================
def crearCalendarioDesdeEstado(ruta: Ruta) -> Dict[int, int]:
    return dict(sorted(ruta.estadoCal.semanaToNodo.items()))


def mostrarCalendario(calendario: Dict[int, int], nodos: List[Nodo]) -> None:
    if not calendario:
        print("Calendario vacío")
        return

    print("\n" + "=" * 105)
    print("CALENDARIO DE CARRERAS F1")
    print("=" * 105)
    print(f"{'#':>3} | {'Semana':>6} | {'Circuito':40} | {'Ciudad':20} | {'País':15}")
    print("-" * 105)

    beneficioFinal = 0
    for idx, semana in enumerate(sorted(calendario.keys()), 1):
        nodoId = calendario[semana]
        nodo = nodos[nodoId]
        beneficioFinal += nodo.recompensa
        print(
            f"{idx:3d} | {semana:6d} | {nodo.nombre:40} | "
            f"{nodo.ciudad:20} | {nodo.pais:15}"
        )

    print("=" * 105)
    print(f"Total de carreras programadas: {len(calendario)}")
    print(f"BENEFICIO FINAL (calendario real): {beneficioFinal}")
    print("=" * 105 + "\n")


# ============================================================
# PREPARACIÓN DEL PROBLEMA
# ============================================================
def prepararDatosProblema() -> DatosProblema:
    nodosBase = leerCircuitosCsv(RUTA_CSV_CIRCUITOS)
    if not nodosBase:
        raise SystemExit("No se pudieron cargar circuitos base")

    gestor = GestorDistancias(nodosBase)

    origenCoord = obtenerCoordCircuito(nodosBase, NOMBRE_CIRCUITO_ORIGEN)
    destinoCoord = obtenerCoordCircuito(nodosBase, NOMBRE_CIRCUITO_DESTINO)

    nodosClientes = expandirNodosPorSemana(nodosBase, numSemanas=NUM_SEMANAS)
    nodos = agregarNodosEspeciales(nodosClientes, origenCoord, destinoCoord)

    origenId = 0
    destinoId = len(nodos) - 1

    print(f"\nTotal de nodos en el problema: {len(nodos)}")

    origenGlobalIds = [
        idx
        for idx, nodo in enumerate(nodos)
        if nodo.circuitoBaseNombre == NOMBRE_CIRCUITO_ORIGEN
        and nodo.semanaAsignada is not None
    ]
    destinoGlobalIds = [
        idx
        for idx, nodo in enumerate(nodos)
        if nodo.circuitoBaseNombre == NOMBRE_CIRCUITO_DESTINO
        and nodo.semanaAsignada is not None
    ]

    if not origenGlobalIds or not destinoGlobalIds:
        raise SystemExit("No se encontraron nodos expandidos para origen/destino")

    mandInicioId = min(
        origenGlobalIds,
        key=lambda nodoId: nodos[nodoId].semanaAsignada,  # type: ignore[arg-type]
    )
    mandFinId = max(
        destinoGlobalIds,
        key=lambda nodoId: nodos[nodoId].semanaAsignada,  # type: ignore[arg-type]
    )

    semanaMin = nodos[mandInicioId].semanaAsignada
    semanaMax = nodos[mandFinId].semanaAsignada

    if semanaMin is None or semanaMax is None:
        raise SystemExit("Error al determinar las semanas obligatorias")

    print("\n" + "=" * 60)
    print("RESTRICCIÓN NEGOCIO: PRIMERA Y ÚLTIMA CARRERA FIJAS")
    print("=" * 60)
    print(
        f"Primera carrera obligatoria: {NOMBRE_CIRCUITO_ORIGEN} "
        f"en semana {semanaMin} (node_id={mandInicioId})"
    )
    print(
        f"Última carrera obligatoria: {NOMBRE_CIRCUITO_DESTINO} "
        f"en semana {semanaMax} (node_id={mandFinId})"
    )
    print("=" * 60)

    tiempo = crearMatrizTiemposOptimizada(nodos, gestor, VEL)

    elegiblesIds: List[int] = []
    for clienteId in range(1, len(nodos) - 1):
        nodo = nodos[clienteId]
        semana = nodo.semanaAsignada

        if semana is None:
            continue
        if not (semanaMin < semana < semanaMax):
            continue
        if nodo.circuitoBaseNombre in {
            NOMBRE_CIRCUITO_ORIGEN,
            NOMBRE_CIRCUITO_DESTINO,
        }:
            continue

        elegiblesIds.append(clienteId)

    customerIds = [mandInicioId, mandFinId] + elegiblesIds

    return DatosProblema(
        nodos=nodos,
        tiempo=tiempo,
        origenId=origenId,
        destinoId=destinoId,
        mandInicioId=mandInicioId,
        mandFinId=mandFinId,
        semanaMin=semanaMin,
        semanaMax=semanaMax,
        elegiblesIds=elegiblesIds,
        customerIds=customerIds,
        maxFines=MAX_FINES_SEGUIDOS,
        maxCarreras=NUM_CARRERAS,
        tMax=T_MAX,
    )


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    datos = prepararDatosProblema()

    mejorRuta, alpha = vnsDeterminista(datos)

    print("\n" + "=" * 70)
    print("MEJOR SOLUCIÓN DETERMINISTA")
    print("=" * 70)
    print(f"Alpha fijo usado en VNS: {alpha}")
    print(mejorRuta)

    calendario = crearCalendarioDesdeEstado(mejorRuta)

    semanas = sorted(calendario.keys())
    if not semanas or semanas[0] != datos.semanaMin or semanas[-1] != datos.semanaMax:
        print(
            "\nADVERTENCIA: el calendario no respeta primera/última "
            "semana obligatoria."
        )

    mostrarCalendario(calendario, datos.nodos)


if __name__ == "__main__":
    main()