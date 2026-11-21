import type { Esquema, ErrorCampo, Prediccion, Salud, Tarea } from "../tipos";

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? "";

export class ErrorApi extends Error {
  errores: ErrorCampo[];
  constructor(mensaje: string, errores: ErrorCampo[] = []) {
    super(mensaje);
    this.name = "ErrorApi";
    this.errores = errores;
  }
}

function esLista(valor: unknown): valor is ErrorCampo[] {
  return Array.isArray(valor);
}

async function pedir<T>(ruta: string, opciones?: RequestInit): Promise<T> {
  let respuesta: Response;
  try {
    respuesta = await fetch(`${BASE}${ruta}`, {
      headers: { "Content-Type": "application/json" },
      ...opciones,
    });
  } catch {
    throw new ErrorApi("No se pudo conectar con el servidor. Revisa tu conexión e inténtalo de nuevo.");
  }

  if (!respuesta.ok) {
    let detalle: unknown = null;
    try {
      const cuerpo = (await respuesta.json()) as { detail?: unknown };
      detalle = cuerpo.detail ?? null;
    } catch {
      detalle = null;
    }
    if (detalle && typeof detalle === "object" && !Array.isArray(detalle)) {
      const d = detalle as { errores?: unknown; mensaje?: unknown };
      if (esLista(d.errores)) {
        const n = d.errores.length;
        throw new ErrorApi(
          n === 1 ? "Hay 1 campo por corregir." : `Hay ${n} campos por corregir.`,
          d.errores
        );
      }
      if (typeof d.mensaje === "string" && respuesta.status < 500) throw new ErrorApi(d.mensaje);
    }
    if (typeof detalle === "string" && respuesta.status < 500) throw new ErrorApi(detalle);
    console.error("Error del servidor", respuesta.status, detalle);
    throw new ErrorApi("No se pudo calcular la predicción en este momento. Inténtalo de nuevo más tarde.");
  }
  return (await respuesta.json()) as T;
}

export function obtenerEsquema(tarea: Tarea): Promise<Esquema> {
  return pedir<Esquema>(`/api/esquemas/${tarea}`);
}

export function obtenerSalud(): Promise<Salud> {
  return pedir<Salud>("/api/salud");
}

export function predecir(tarea: Tarea, datos: Record<string, unknown>): Promise<Prediccion> {
  return pedir<Prediccion>(`/api/predecir/${tarea}`, {
    method: "POST",
    body: JSON.stringify({ datos }),
  });
}
