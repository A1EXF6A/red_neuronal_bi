import { useRef, useState } from "react";
import type { FormEvent } from "react";
import type { CampoEsquema, Esquema, Prediccion, ValorCampo } from "../tipos";
import { ErrorApi, predecir } from "../api/cliente";
import Campo from "./Campo";
import ResultadoPrediccion from "./ResultadoPrediccion";

type Valores = Record<string, ValorCampo>;

// Obtiene valores iniciales
function valoresIniciales(esquema: Esquema): Valores {
  const valores: Valores = {};
  for (const seccion of esquema.secciones) {
    for (const campo of seccion.campos) {
      valores[campo.nombre] = campo.tipo === "booleano" ? Boolean(campo.defecto ?? false) : "";
    }
  }
  return valores;
}

// Obtiene valores de ejemplo
function valoresDeEjemplo(esquema: Esquema): Valores {
  const valores = valoresIniciales(esquema);
  for (const seccion of esquema.secciones) {
    for (const campo of seccion.campos) {
      if (campo.ejemplo === undefined) continue;
      valores[campo.nombre] = campo.tipo === "booleano" ? Boolean(campo.ejemplo) : String(campo.ejemplo);
    }
  }
  return valores;
}

// Comprueba la visibilidad de un campo
function esVisible(campo: CampoEsquema, valores: Valores): boolean {
  if (!campo.mostrar_si) return true;
  return valores[campo.mostrar_si.campo] === campo.mostrar_si.valor;
}

/** Datos del formulario. */
// Prepara los datos del formulario
function armarDatos(esquema: Esquema, valores: Valores): Record<string, unknown> {
  const datos: Record<string, unknown> = {};
  for (const seccion of esquema.secciones) {
    for (const campo of seccion.campos) {
      if (!esVisible(campo, valores)) continue;
      const valor = valores[campo.nombre];
      if (campo.tipo === "booleano") {
        datos[campo.nombre] = Boolean(valor);
        continue;
      }
      const texto = String(valor ?? "").trim();
      if (texto === "") continue;
      if (campo.tipo === "numero" || campo.tipo === "entero") {
        const numero = Number(texto.replace(",", "."));
        datos[campo.nombre] = Number.isNaN(numero) ? texto : numero; // convertir numero
      } else {
        datos[campo.nombre] = texto;
      }
    }
  }
  return datos;
}

// Renderiza el formulario
export default function FormularioRed({ esquema }: { esquema: Esquema }) {
  const [valores, setValores] = useState<Valores>(() => valoresIniciales(esquema));
  const [errores, setErrores] = useState<Record<string, string>>({});
  const [resultado, setResultado] = useState<Prediccion | null>(null);
  const [cargando, setCargando] = useState(false);
  const [errorGeneral, setErrorGeneral] = useState<string | null>(null);
  const refFormulario = useRef<HTMLFormElement | null>(null);
  const refResultado = useRef<HTMLDivElement | null>(null);

  // Actualiza un campo
  function cambiar(nombre: string, valor: ValorCampo) {
    setValores((previo) => ({ ...previo, [nombre]: valor }));
    if (errores[nombre]) {
      setErrores((previo) => {
        const copia = { ...previo };
        delete copia[nombre];
        return copia;
      });
    }
  }

  // Limpia el formulario
  function limpiar() {
    setValores(valoresIniciales(esquema));
    setErrores({});
    setErrorGeneral(null);
    setResultado(null);
  }

  // Carga un ejemplo
  function cargarEjemplo() {
    setValores(valoresDeEjemplo(esquema));
    setErrores({});
    setErrorGeneral(null);
    setResultado(null);
  }

  // Enfoca el primer error
  function enfocarPrimerError(campos: string[]) {
    window.setTimeout(() => {
      for (const nombre of campos) {
        const contenedor = refFormulario.current?.querySelector<HTMLElement>(
          `[data-campo="${nombre}"]`
        );
        const control = contenedor?.querySelector<HTMLElement>("input, button");
        if (control) {
          control.focus();
          return;
        }
      }
    }, 0);
  }

  // Envia el formulario
  async function enviar(evento: FormEvent<HTMLFormElement>) {
    evento.preventDefault();
    setCargando(true);
    setErrorGeneral(null);
    setErrores({});
    try {
      const prediccion = await predecir(esquema.tarea, armarDatos(esquema, valores));
      setResultado(prediccion);
      window.requestAnimationFrame(() => {
        const pantallaChica = window.matchMedia("(max-width: 959px)").matches;
        const sinMovimiento = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        if (pantallaChica) {
          refResultado.current?.scrollIntoView({
            behavior: sinMovimiento ? "auto" : "smooth",
            block: "start",
          });
        }
      });
    } catch (error) {
      setResultado(null);
      if (error instanceof ErrorApi) {
        const porCampo: Record<string, string> = {};
        for (const e of error.errores) porCampo[e.campo] = e.mensaje;
        setErrores(porCampo);
        setErrorGeneral(error.message);
        enfocarPrimerError(error.errores.map((e) => e.campo));
      } else {
        setErrorGeneral("Ocurrió un error inesperado al predecir.");
      }
    } finally {
      setCargando(false);
    }
  }

  return (
    <div className="red">
      <form className="papel formulario" onSubmit={enviar} noValidate ref={refFormulario}>
        <header className="formulario-cabecera">
          <h2>{esquema.titulo}</h2>
          <p>{esquema.descripcion}</p>
        </header>

        {esquema.secciones.map((seccion) => {
          const campos = seccion.campos.filter((c) => esVisible(c, valores));
          return (
            <section className="seccion" key={seccion.id} aria-labelledby={`${esquema.tarea}-${seccion.id}`}>
              <h3 id={`${esquema.tarea}-${seccion.id}`}>{seccion.titulo}</h3>
              {seccion.descripcion && <p className="seccion-nota">{seccion.descripcion}</p>}
              <div className="campos">
                {campos.map((campo) => (
                  <Campo
                    key={campo.nombre}
                    campo={campo}
                    valor={valores[campo.nombre]}
                    error={errores[campo.nombre]}
                    onCambio={cambiar}
                  />
                ))}
              </div>
            </section>
          );
        })}

        {errorGeneral && (
          <p className="banner banner-error" role="alert">
            {errorGeneral}
          </p>
        )}

        <div className="acciones">
          <button type="submit" className="boton boton-primario" disabled={cargando}>
            {cargando ? "Calculando…" : "Predecir"}
          </button>
          <button type="button" className="boton boton-secundario" onClick={cargarEjemplo} disabled={cargando}>
            Cargar ejemplo
          </button>
          <button type="button" className="boton boton-texto" onClick={limpiar} disabled={cargando}>
            Limpiar
          </button>
        </div>
      </form>

      <div className="panel-resultado" ref={refResultado}>
        <ResultadoPrediccion esquema={esquema} resultado={resultado} cargando={cargando} />
      </div>
    </div>
  );
}
