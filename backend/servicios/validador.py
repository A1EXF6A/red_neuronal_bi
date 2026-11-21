"""Validación de los datos escritos en el formulario.

No depende de FastAPI ni de TensorFlow. Convierte texto/números/fechas al tipo correcto,
comprueba límites duros (valores imposibles) y reconoce las categorías vistas en el
entrenamiento ignorando mayúsculas, espacios y acentos. Las categorías NUEVAS se aceptan
(no se rechazan): ver `avisos_categorias_nuevas`.
"""
from __future__ import annotations

import math
import re
import unicodedata
from datetime import date


LARGO_MAXIMO_TEXTO = 100


class ErrorValidacion(Exception):
    def __init__(self, errores: list[dict]):
        super().__init__("Datos inválidos")
        self.errores = errores


# ---------------------------------------------------------------- utilidades de texto
def _sin_acentos(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))


def clave_texto(texto) -> str:
    """'  São  paulo ' -> 'sao paulo'."""
    return re.sub(r"\s+", " ", _sin_acentos(str(texto)).strip()).casefold()


def campos_planos(esquema: dict) -> list[dict]:
    return [campo for seccion in esquema["secciones"] for campo in seccion["campos"]]


def _vacio(valor) -> bool:
    return valor is None or (isinstance(valor, str) and valor.strip() == "")


# ---------------------------------------------------------------- conversión por tipo
_VERDADERO = {"true", "1", "si", "sí", "s", "yes", "y", "verdadero"}
_FALSO = {"false", "0", "no", "n", "falso"}


def _a_numero(valor) -> float:
    if isinstance(valor, bool):
        raise ValueError("Debe ser un número")
    if isinstance(valor, (int, float)):
        numero = float(valor)
    else:
        texto = str(valor).strip().replace(",", ".")
        try:
            numero = float(texto)
        except ValueError:
            raise ValueError("Debe ser un número (ejemplo: 12.5)")
    if math.isnan(numero) or math.isinf(numero):
        raise ValueError("Debe ser un número finito")
    return numero


def _a_booleano(valor) -> bool:
    if isinstance(valor, bool):
        return valor
    texto = str(valor).strip().casefold()
    if texto in _VERDADERO:
        return True
    if texto in _FALSO:
        return False
    raise ValueError("Debe ser Sí o No")


def _a_fecha(valor) -> date:
    texto = str(valor).strip()[:10]
    try:
        return date.fromisoformat(texto)
    except ValueError:
        raise ValueError("Fecha inválida; usa el formato AAAA-MM-DD")


def _convertir(campo: dict, bruto, indice_categorias: dict):
    tipo = campo["tipo"]
    if tipo == "booleano":
        return _a_booleano(bruto)
    if tipo == "fecha":
        return _a_fecha(bruto)
    if tipo in ("numero", "entero"):
        numero = _a_numero(bruto)
        if tipo == "entero":
            if numero != int(numero):
                raise ValueError("Debe ser un número entero")
            numero = int(numero)
        minimo, maximo = campo.get("min"), campo.get("max")
        if minimo is not None and numero < minimo:
            raise ValueError(f"Debe ser mayor o igual a {minimo}")
        if maximo is not None and numero > maximo:
            raise ValueError(f"Debe ser menor o igual a {maximo}")
        return numero
    if tipo == "texto":
        texto = re.sub(r"\s+", " ", str(bruto).strip())
        if len(texto) > LARGO_MAXIMO_TEXTO:
            raise ValueError(f"Máximo {LARGO_MAXIMO_TEXTO} caracteres")
        indice = indice_categorias.get(campo["nombre"], {})
        # Conocido en el entrenamiento -> nombre canónico (ignora mayúsculas/acentos/espacios).
        # Nuevo -> se acepta tal cual; el OneHotEncoder (handle_unknown="ignore") lo codifica
        # con ceros y el modelo predice con el resto de variables.
        return indice.get(clave_texto(texto), texto)
    raise ValueError(f"Tipo de campo desconocido: {tipo}")


# ---------------------------------------------------------------- validación completa
def validar(esquema: dict, datos: dict, indice_categorias: dict) -> tuple[dict, list[dict]]:
    """Devuelve (valores, errores). `valores[nombre]` es None si el campo quedó vacío u oculto."""
    campos = campos_planos(esquema)
    analizados: dict[str, tuple] = {}

    # 1) Convierte todos los campos (la visibilidad se decide después).
    for campo in campos:
        nombre = campo["nombre"]
        bruto = datos.get(nombre)
        if _vacio(bruto):
            if campo["tipo"] == "booleano" and "defecto" in campo:
                analizados[nombre] = (bool(campo["defecto"]), None)
            else:
                analizados[nombre] = (None, None)
            continue
        try:
            analizados[nombre] = (_convertir(campo, bruto, indice_categorias), None)
        except ValueError as exc:
            analizados[nombre] = (None, str(exc))

    # 2) Aplica visibilidad (mostrar_si) y obligatoriedad.
    valores: dict = {}
    errores: list[dict] = []
    for campo in campos:
        nombre = campo["nombre"]
        valor, error = analizados[nombre]
        condicion = campo.get("mostrar_si")
        if condicion and analizados.get(condicion["campo"], (None, None))[0] != condicion["valor"]:
            valores[nombre] = None  # campo oculto: se ignora
            continue
        if error:
            errores.append({"campo": nombre, "mensaje": error})
            continue
        if valor is None and campo.get("requerido", False):
            errores.append({"campo": nombre, "mensaje": "Campo obligatorio"})
            continue
        valores[nombre] = valor
    return valores, errores


# ---------------------------------------------------------------- avisos (no bloquean)
def avisos_de_rango(esquema: dict, valores: dict, rangos: dict, rango_fechas: list | None) -> list[str]:
    """Advierte cuando un valor queda fuera de lo visto en entrenamiento (extrapolación)."""
    avisos: list[str] = []
    for campo in campos_planos(esquema):
        valor = valores.get(campo["nombre"])
        if valor is None:
            continue
        if campo["tipo"] in ("numero", "entero"):
            columna = campo.get("rango_ref", campo["nombre"])
            if columna in rangos:
                bajo, alto = rangos[columna]
                if valor < bajo or valor > alto:
                    avisos.append(
                        f"«{campo['etiqueta']}» = {valor} es muy distinto de lo habitual "
                        f"(lo normal va de {bajo:g} a {alto:g}); el resultado puede ser menos preciso."
                    )
        elif campo["tipo"] == "fecha" and campo["nombre"] == "fecha" and rango_fechas:
            desde, hasta = date.fromisoformat(rango_fechas[0]), date.fromisoformat(rango_fechas[1])
            if valor < desde or valor > hasta:
                avisos.append(
                    f"La fecha es muy distinta de las habituales (de {desde} a {hasta}); "
                    "el resultado puede ser menos preciso."
                )
    return avisos


def es_categoria_nueva(campo_nombre: str, valor, indice_categorias: dict) -> bool:
    """True si el texto no coincide con ninguna categoría vista en el entrenamiento."""
    conocidas = set(indice_categorias.get(campo_nombre, {}).values())
    return valor is not None and valor not in conocidas


def avisos_categorias_nuevas(esquema: dict, valores: dict, indice_categorias: dict) -> list[str]:
    """Advierte (sin bloquear) de textos que el modelo no vio al entrenar."""
    avisos: list[str] = []
    for campo in campos_planos(esquema):
        if campo["tipo"] != "texto":
            continue
        valor = valores.get(campo["nombre"])
        if es_categoria_nueva(campo["nombre"], valor, indice_categorias):
            avisos.append(
                f"«{campo['etiqueta']}»: «{valor}» es un valor nuevo y no hay datos previos sobre él; "
                "el resultado se basa en el resto de la información."
            )
    return avisos
