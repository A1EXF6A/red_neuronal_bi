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
 * Los campos de texto solo aceptan letras; los numéricos solo aceptan dígitos.
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
  const esTexto = campo.tipo === "texto";
  const esNumerico = campo.tipo === "entero" || campo.tipo === "numero";
  const marcador =
    !esFecha && campo.ejemplo !== undefined ? `Ej.: ${String(campo.ejemplo)}` : undefined;

  const filtrarTexto = (valorEntrada: string): string =>
    valorEntrada.replace(/[^a-zA-ZáéíóúñÁÉÍÓÚÑüÜ\s]/g, "");

  const filtrarNumero = (valorEntrada: string): string =>
    valorEntrada.replace(/[^0-9.,-]/g, "");

  const filtrar = (valorEntrada: string): string => {
    if (esTexto) return filtrarTexto(valorEntrada);
    if (esNumerico) return filtrarNumero(valorEntrada);
    return valorEntrada;
  };

  const manejarCambio = (e: React.ChangeEvent<HTMLInputElement>) => {
    const limpio = filtrar(e.target.value);
    onCambio(campo.nombre, limpio);
  };

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
          pattern={esTexto ? "[a-zA-ZáéíóúñÁÉÍÓÚÑüÜ\\s]*" : esNumerico ? "[0-9.,-]*" : undefined}
          placeholder={marcador}
          value={String(valor)}
          autoComplete="off"
          spellCheck={false}
          aria-invalid={error ? true : undefined}
          aria-describedby={describedby}
          onChange={manejarCambio}
        />
        {campo.unidad && <span className="unidad">{campo.unidad}</span>}
      </div>
      {ayuda}
      {mensajeError}
    </div>
  );
}
