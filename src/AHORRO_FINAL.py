import csv
import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ============================================================
# PARÁMETROS PERSONALIZABLES
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
ALFA: float = 0.7
RUTA_CSV_CIRCUITOS: str = "circuitos_f1_beneficios_disponible_simulados.csv"

# ============================================================
# NODO
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
        self.semanasProhibidas: List[int] = (
            semanasProhibidas if semanasProhibidas else []
        )
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


# ============================================================
# ESTADO INCREMENTAL DE CALENDARIO
# ============================================================
@dataclass
class EstadoCalendario:
    """Estado incremental de calendario con trazabilidad exacta semana→nodo."""

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


# ============================================================
# RUTA
# ============================================================
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
        self.nodosVisitados: Set[int] = {0}
        self.circuitosBaseVisitados: Set[str] = set()
        self.estadoCal: EstadoCalendario = EstadoCalendario.vacio()
        self.tieneMandatorias: bool = False

    def esValida(self, tMax: float) -> bool:
        return self.tiempoTotal <= tMax

    def __repr__(self) -> str:
        return (
            f"Ruta {self.id}: nodos={len(self.nodos) - 2}, "
            f"T={self.tiempoTotal:.2f}, Rbruta={self.recompensaTotal}, "
            f"Carreras={self.estadoCal.numCarreras}, "
            f"BenefReal={self.estadoCal.beneficioReal}, "
            f"Mandatorias={self.tieneMandatorias}"
        )


# ============================================================
# GESTOR DE DISTANCIAS
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
                        int(semana.strip())
                        for semana in raw.split(",")
                        if semana.strip().isdigit()
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

        print(f"Se cargaron {len(nodosBase)} circuitos base del CSV")

    except FileNotFoundError:
        print("Error: Archivo de circuitos no encontrado")

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


# ============================================================
# MATRIZ DE TIEMPOS
# ============================================================
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
# RESTRICCIÓN: MÁXIMO DE FINES CONSECUTIVOS
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
# SAVINGS
# ============================================================
def calcularSavings(
    i: int,
    j: int,
    alfa: float,
    nodosList: List[Nodo],
    tiempoMat: List[List[float]],
    origenId: int,
    destinoId: int,
) -> float:
    reward = nodosList[i].recompensa + nodosList[j].recompensa
    timeSaved = (
        tiempoMat[i][destinoId] + tiempoMat[origenId][j] - tiempoMat[i][j]
    )
    return alfa * (timeSaved) + (1.0 - alfa) * reward


# ============================================================
# FUSIÓN
# ============================================================
def fusionClarkeWright(
    rutaI: Ruta,
    rutaJ: Ruta,
    maxFinesSeguidos: int,
    nodos: List[Nodo],
    tiempoMat: List[List[float]],
    tMax: float,
    maxCarreras: int,
    origenId: int,
    destinoId: int,
) -> Optional[Ruta]:
    if not rutaI.circuitosBaseVisitados.isdisjoint(rutaJ.circuitosBaseVisitados):
        return None

    nuevoEstadoCal = fusionarEstadosCalendario(
        rutaI.estadoCal,
        rutaJ.estadoCal,
        maxFines=maxFinesSeguidos,
        maxCarreras=maxCarreras,
    )
    if nuevoEstadoCal is None:
        return None

    clientesSet = set(rutaI.nodos[1:-1]) | set(rutaJ.nodos[1:-1])
    clientes = sorted(
        clientesSet,
        key=lambda nodoId: (
            nodos[nodoId].semanaAsignada
            if nodos[nodoId].semanaAsignada is not None
            else 999
        ),
    )

    nuevaRuta = Ruta(max(rutaI.id, rutaJ.id) + 100)
    nuevaRuta.nodos = [origenId] + clientes + [destinoId]
    nuevaRuta.nodosVisitados = set(nuevaRuta.nodos)
    nuevaRuta.recompensaTotal = rutaI.recompensaTotal + rutaJ.recompensaTotal
    nuevaRuta.circuitosBaseVisitados = (
        rutaI.circuitosBaseVisitados | rutaJ.circuitosBaseVisitados
    )
    nuevaRuta.estadoCal = nuevoEstadoCal
    nuevaRuta.tieneMandatorias = rutaI.tieneMandatorias or rutaJ.tieneMandatorias

    nuevaRuta.tiempoTotal = sum(
        tiempoMat[nuevaRuta.nodos[k]][nuevaRuta.nodos[k + 1]]
        for k in range(len(nuevaRuta.nodos) - 1)
    )

    if nuevaRuta.tiempoTotal > tMax:
        return None

    return nuevaRuta


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
# CONSTRUCCIÓN DE RUTAS INICIALES
# ============================================================
def _rutaSingleton(
    clienteId: int,
    nodos: List[Nodo],
    origenId: int,
    destinoId: int,
    tiempoMat: List[List[float]],
    idRuta: int,
    tMax: float,
    semanaMin: int,
    semanaMax: int,
) -> Optional[Ruta]:
    nodo = nodos[clienteId]
    semana = nodo.semanaAsignada

    if semana is None:
        return None

    if not (semanaMin < semana < semanaMax):
        return None

    tiempo = tiempoMat[origenId][clienteId] + tiempoMat[clienteId][destinoId]
    if tiempo > tMax:
        return None

    ruta = Ruta(idRuta)
    ruta.nodos = [origenId, clienteId, destinoId]
    ruta.nodosVisitados = {origenId, clienteId, destinoId}
    ruta.recompensaTotal = nodo.recompensa
    ruta.tiempoTotal = tiempo

    if nodo.circuitoBaseNombre:
        ruta.circuitosBaseVisitados.add(nodo.circuitoBaseNombre)

    ruta.estadoCal = EstadoCalendario.desdeNodo(semana, clienteId, nodo.recompensa)
    ruta.tieneMandatorias = False

    return ruta


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
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

    origenNodeId = min(
        origenGlobalIds, key=lambda nodoId: nodos[nodoId].semanaAsignada
    )
    destinoNodeId = max(
        destinoGlobalIds, key=lambda nodoId: nodos[nodoId].semanaAsignada
    )

    semanaMin = nodos[origenNodeId].semanaAsignada  # type: ignore[assignment]
    semanaMax = nodos[destinoNodeId].semanaAsignada  # type: ignore[assignment]

    print("\n" + "=" * 60)
    print("RESTRICCIÓN NEGOCIO: PRIMERA Y ÚLTIMA CARRERA FIJAS")
    print("=" * 60)
    print(
        f"Primera carrera obligatoria: {NOMBRE_CIRCUITO_ORIGEN} "
        f"en semana {semanaMin} (node_id={origenNodeId})"
    )
    print(
        f"Última carrera obligatoria: {NOMBRE_CIRCUITO_DESTINO} "
        f"en semana {semanaMax} (node_id={destinoNodeId})"
    )
    print("=" * 60)

    tiempo = crearMatrizTiemposOptimizada(nodos, gestor, VEL)

    seed = Ruta(0)
    seed.nodos = [origenId, origenNodeId, destinoNodeId, destinoId]
    seed.nodosVisitados = set(seed.nodos)
    seed.circuitosBaseVisitados = {
        nodos[origenNodeId].circuitoBaseNombre or NOMBRE_CIRCUITO_ORIGEN,
        nodos[destinoNodeId].circuitoBaseNombre or NOMBRE_CIRCUITO_DESTINO,
    }
    seed.recompensaTotal = (
        nodos[origenNodeId].recompensa + nodos[destinoNodeId].recompensa
    )

    estadoOrigen = EstadoCalendario.desdeNodo(
        semanaMin, origenNodeId, nodos[origenNodeId].recompensa
    )
    estadoDestino = EstadoCalendario.desdeNodo(
        semanaMax, destinoNodeId, nodos[destinoNodeId].recompensa
    )

    estadoSeed = fusionarEstadosCalendario(
        estadoOrigen,
        estadoDestino,
        MAX_FINES_SEGUIDOS,
        NUM_CARRERAS,
    )
    if estadoSeed is None:
        raise SystemExit(
            "Inviable: las carreras obligatorias violan la restricción "
            "de consecutividad o duplican semana"
        )

    seed.estadoCal = estadoSeed
    seed.tieneMandatorias = True
    seed.tiempoTotal = sum(
        tiempo[seed.nodos[k]][seed.nodos[k + 1]]
        for k in range(len(seed.nodos) - 1)
    )

    if seed.tiempoTotal > T_MAX:
        raise SystemExit("Inviable: la ruta seed excede T_MAX")

    rutas: List[Ruta] = [seed]

    for clienteId in range(1, len(nodos) - 1):
        nodo = nodos[clienteId]
        if nodo.circuitoBaseNombre in {
            NOMBRE_CIRCUITO_ORIGEN,
            NOMBRE_CIRCUITO_DESTINO,
        }:
            continue

        ruta = _rutaSingleton(
            clienteId,
            nodos,
            origenId,
            destinoId,
            tiempo,
            len(rutas),
            T_MAX,
            semanaMin,
            semanaMax,
        )
        if ruta is not None:
            rutas.append(ruta)

    print(
        f"\nSolución inicial: {len(rutas)} rutas "
        f"(1 seed obligatoria + singletons)"
    )

    print("\nCalculando savings...")
    nClientes = len(nodos) - 2
    savingsList: List[Tuple[float, int, int]] = []

    for i in range(1, nClientes + 1):
        for j in range(i + 1, nClientes + 1):
            savingsList.append(
                (
                    calcularSavings(i, j, ALFA, nodos, tiempo, origenId, destinoId),
                    i,
                    j,
                )
            )
            savingsList.append(
                (
                    calcularSavings(j, i, ALFA, nodos, tiempo, origenId, destinoId),
                    j,
                    i,
                )
            )

    savingsList.sort(reverse=True, key=lambda item: item[0])
    print(f"Savings calculados: {len(savingsList)}")

    nodoARuta: Dict[int, Ruta] = {}
    for ruta in rutas:
        for nodoId in ruta.nodosVisitados:
            nodoARuta[nodoId] = ruta

    print("\n" + "=" * 70)
    print("FUSIÓN: SOLO PERMITIMOS CRECER LA RUTA CON MANDATORIAS")
    print("=" * 70)

    fusiones = 0
    rechazadasNoSeed = 0
    rechazadasBase = 0
    rechazadasCal = 0
    rechazadasTMax = 0

    for _, i, j in savingsList:
        rutaI = nodoARuta.get(i)
        rutaJ = nodoARuta.get(j)

        if not rutaI or not rutaJ or rutaI.id == rutaJ.id:
            continue

        if not (rutaI.tieneMandatorias or rutaJ.tieneMandatorias):
            rechazadasNoSeed += 1
            continue

        if not rutaI.circuitosBaseVisitados.isdisjoint(rutaJ.circuitosBaseVisitados):
            rechazadasBase += 1
            continue

        nuevaRuta = fusionClarkeWright(
            rutaI,
            rutaJ,
            MAX_FINES_SEGUIDOS,
            nodos,
            tiempo,
            T_MAX,
            NUM_CARRERAS,
            origenId,
            destinoId,
        )

        if nuevaRuta is None:
            rechazadasCal += 1
            continue

        if nuevaRuta.tiempoTotal > T_MAX:
            rechazadasTMax += 1
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

        fusiones += 1
        if fusiones <= 5:
            print(f"\nFusión {fusiones}: R{rutaI.id} + R{rutaJ.id} -> R{nuevaRuta.id}")
            print(
                f" Carreras={nuevaRuta.estadoCal.numCarreras}, "
                f"BenefReal={nuevaRuta.estadoCal.beneficioReal}, "
                f"RachaMax={nuevaRuta.estadoCal.rachaMax}"
            )

    print("\n" + "=" * 70)
    print("ESTADÍSTICAS")
    print("=" * 70)
    print(f"Fusiones realizadas : {fusiones}")
    print(f"Rechazos (no implican ruta seed) : {rechazadasNoSeed}")
    print(f"Rechazos (circuito base repetido) : {rechazadasBase}")
    print(f"Rechazos (calendario/racha/semanas) : {rechazadasCal}")

    rutasMandatorias = [ruta for ruta in rutas if ruta.tieneMandatorias]
    if not rutasMandatorias:
        raise SystemExit(
            "Error: no quedó ninguna ruta con las carreras obligatorias"
        )

    rutasMandatorias.sort(
        key=lambda ruta: ruta.estadoCal.beneficioReal,
        reverse=True,
    )
    solucionFinal = rutasMandatorias[:NUM_VEHICULOS]

    print("\n" + "=" * 70)
    print("SOLUCIÓN FINAL")
    print("=" * 70)
    for idx, ruta in enumerate(solucionFinal, 1):
        print(f"Vehículo {idx}: {ruta}")

    calendario = crearCalendarioDesdeEstado(solucionFinal[0])

    semanas = sorted(calendario.keys())
    if not semanas or semanas[0] != semanaMin or semanas[-1] != semanaMax:
        print(
            "\nADVERTENCIA: el calendario no respeta primera/última semana "
            "obligatoria (esto no debería ocurrir)."
        )

    mostrarCalendario(calendario, nodos)