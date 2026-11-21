import { useId } from "react";
import type { CampoEsquema, ValorCampo } from "../tipos";

interface Props {
  campo: CampoEsquema;
  valor: ValorCampo;
  error?: string;
  onCambio: (nombre: string, valor: ValorCampo) => void;
}

/**
 * Un campo del formulario. Solo hay entradas de texto, número y fecha, e interruptores Sí/No.
 * En los campos de texto, `valores_validos` son solo sugerencias (datalist): se puede escribir
 * cualquier valor.
 */
export default function Campo({ campo, valor, error, onCambio }: Props) {
  const id = useId();
  const idAyuda = `${id}-ayuda`;
  const idError = `${id}-error`;
  const describedby =
    [campo.ayuda ? idAyuda : "", error ? idError : ""].filter(Boolean).join(" ") || undefined;

  const ayuda = campo.ayuda ? (
    <p id={idAyuda} className="ayuda">
      {campo.ayuda}
    </p>
  ) : null;
  const mensajeError = error ? (
    <p id={idError} className="mensaje-error" role="alert">
      {error}
    </p>
  ) : null;

  if (campo.tipo === "booleano") {
    const activo = Boolean(valor);
    return (
      <div className="campo campo-interruptor" data-campo={campo.nombre}>
        <div className="interruptor-fila">
          <span id={`${id}-etiqueta`} className="etiqueta">
            {campo.etiqueta}
          </span>
          <button
            type="button"
            role="switch"
            aria-checked={activo}
            aria-labelledby={`${id}-etiqueta`}
            aria-describedby={describedby}
            className={`interruptor${activo ? " activo" : ""}`}
            onClick={() => onCambio(campo.nombre, !activo)}
          >
            <span className="interruptor-pista" aria-hidden="true">
              <span className="interruptor-bola" />
            </span>
            <span className="interruptor-texto">{activo ? "Sí" : "No"}</span>
          </button>
        </div>
        {ayuda}
        {mensajeError}
      </div>
    );
  }

  const esFecha = campo.tipo === "fecha";
  const modo =
    campo.tipo === "entero" ? "numeric" : campo.tipo === "numero" ? "decimal" : undefined;
  const marcador =
    !esFecha && campo.ejemplo !== undefined ? `Ej.: ${String(campo.ejemplo)}` : undefined;
  const sugerencias = campo.tipo === "texto" ? campo.valores_validos ?? [] : [];
  const idLista = `${id}-sugerencias`;

  return (
    <div className={`campo${error ? " con-error" : ""}`} data-campo={campo.nombre}>
      <label htmlFor={id} className="etiqueta">
        {campo.etiqueta}
        {!campo.requerido && <span className="opcional"> (opcional)</span>}
      </label>
      <div className="entrada">
        <input
          id={id}
          name={campo.nombre}
          type={esFecha ? "date" : "text"}
          inputMode={modo}
          list={sugerencias.length > 0 ? idLista : undefined}
          placeholder={marcador}
          value={String(valor)}
          autoComplete="off"
          spellCheck={false}
          aria-invalid={error ? true : undefined}
          aria-describedby={describedby}
          onChange={(e) => onCambio(campo.nombre, e.target.value)}
        />
        {sugerencias.length > 0 && (
          <datalist id={idLista}>
            {sugerencias.map((v) => (
              <option key={v} value={v} />
            ))}
          </datalist>
        )}
        {campo.unidad && <span className="unidad">{campo.unidad}</span>}
      </div>
      {ayuda}
      {mensajeError}
    </div>
  );
}
