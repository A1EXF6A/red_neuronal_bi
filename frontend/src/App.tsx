import { useCallback, useEffect, useState } from "react";
import type { KeyboardEvent } from "react";
import type { Esquema, Salud, Tarea } from "./tipos";
import { ErrorApi, obtenerEsquema, obtenerSalud } from "./api/cliente";
import FormularioRed from "./components/FormularioRed";

const TAREAS: Tarea[] = ["pedidos", "resenas", "precios"];

type Esquemas = Partial<Record<Tarea, Esquema>>;

export default function App() {
  const [activa, setActiva] = useState<Tarea>("pedidos");
  const [esquemas, setEsquemas] = useState<Esquemas>({});
  const [estado, setEstado] = useState<"cargando" | "listo" | "error">("cargando");
  const [mensajeError, setMensajeError] = useState("");
  const [salud, setSalud] = useState<Salud | null>(null);

  const cargar = useCallback(async () => {
    setEstado("cargando");
    try {
      const lista = await Promise.all(TAREAS.map((t) => obtenerEsquema(t)));
      const mapa: Esquemas = {};
      lista.forEach((esquema, i) => {
        mapa[TAREAS[i]] = esquema;
      });
      setEsquemas(mapa);
      setEstado("listo");
      try {
        setSalud(await obtenerSalud());
      } catch {
        setSalud(null);
      }
    } catch (error) {
      setMensajeError(error instanceof ErrorApi ? error.message : "No se pudo cargar la aplicación.");
      setEstado("error");
    }
  }, []);

  useEffect(() => {
    void cargar();
  }, [cargar]);

  function teclaEnPestanas(evento: KeyboardEvent<HTMLDivElement>) {
    const posicion = TAREAS.indexOf(activa);
    let nueva = posicion;
    if (evento.key === "ArrowRight") nueva = (posicion + 1) % TAREAS.length;
    else if (evento.key === "ArrowLeft") nueva = (posicion - 1 + TAREAS.length) % TAREAS.length;
    else if (evento.key === "Home") nueva = 0;
    else if (evento.key === "End") nueva = TAREAS.length - 1;
    else return;
    evento.preventDefault();
    setActiva(TAREAS[nueva]);
    document.getElementById(`pestana-${TAREAS[nueva]}`)?.focus();
  }

  const redesSinModelo = salud
    ? TAREAS.filter((t) => !salud.redes[t]?.modelo).map((t) => ({ tarea: t }))
    : [];

  return (
    <div className="aplicacion">
      <header className="encabezado">
        <h1>GlobalBite Delivery</h1>
        <p className="encabezado-texto">
          Completa los datos de tu caso y obtén una predicción al instante. Elige arriba qué quieres
          estimar: la demora de un pedido, la satisfacción del cliente o un cambio de precio.
        </p>
      </header>

      {estado === "cargando" && <p className="banner banner-info">Conectando con el servidor…</p>}

      {estado === "error" && (
        <div className="banner banner-error" role="alert">
          <p>{mensajeError}</p>
          <button type="button" className="boton boton-secundario" onClick={() => void cargar()}>
            Reintentar
          </button>
        </div>
      )}

      {redesSinModelo.length > 0 && (
        <div className="banner banner-aviso" role="alert">
          {redesSinModelo.map((r) => (
            <p key={r.tarea}>
              La predicción de «{esquemas[r.tarea]?.nombre_corto ?? r.tarea}» no está disponible por el momento.
            </p>
          ))}
        </div>
      )}

      {estado === "listo" && (
        <>
          <div className="pestanas" role="tablist" aria-label="Qué quieres predecir" onKeyDown={teclaEnPestanas}>
            {TAREAS.map((tarea) => {
              const esquema = esquemas[tarea];
              if (!esquema) return null;
              const seleccionada = tarea === activa;
              return (
                <button
                  key={tarea}
                  id={`pestana-${tarea}`}
                  type="button"
                  role="tab"
                  aria-selected={seleccionada}
                  aria-controls={`panel-${tarea}`}
                  tabIndex={seleccionada ? 0 : -1}
                  className={`pestana${seleccionada ? " activa" : ""}`}
                  onClick={() => setActiva(tarea)}
                >
                  <span className="pestana-nombre">{esquema.nombre_corto}</span>
                  <span className="pestana-dato">{esquema.titulo}</span>
                </button>
              );
            })}
          </div>

          {TAREAS.map((tarea) => {
            const esquema = esquemas[tarea];
            if (!esquema) return null;
            return (
              <div
                key={tarea}
                id={`panel-${tarea}`}
                role="tabpanel"
                aria-labelledby={`pestana-${tarea}`}
                hidden={tarea !== activa}
              >
                <FormularioRed esquema={esquema} />
              </div>
            );
          })}
        </>
      )}
    </div>
  );
}
