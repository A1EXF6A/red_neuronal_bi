import type { Esquema, Prediccion } from "../tipos";

interface Props {
  esquema: Esquema;
  resultado: Prediccion | null;
  cargando: boolean;
}

// Formatea un porcentaje
function porcentaje(p: number): string {
  return `${(p * 100).toLocaleString("es", { minimumFractionDigits: 1, maximumFractionDigits: 1 })} %`;
}

// Muestra el resultado
export default function ResultadoPrediccion({ esquema, resultado, cargando }: Props) {
  return (
    <div className="ticket-envoltura" aria-live="polite">
      <div className="ticket">
        {!resultado && (
          <>
            <h2 className="ticket-titulo">{cargando ? "Calculando…" : "Aquí aparecerá la predicción"}</h2>
            <p className="ticket-texto">{esquema.momento_prediccion}</p>
            <div className="ticket-corte" />
            <h3 className="ticket-subtitulo">Posibles resultados</h3>
            <ul className="leyenda">
              {esquema.clases.map((c) => (
                <li key={c.clase}>
                  <span>
                    <strong>{c.nombre}</strong>, {c.detalle}
                  </span>
                </li>
              ))}
            </ul>
          </>
        )}

        {resultado && (
          <>
            <p className="ticket-modelo">{esquema.titulo}</p>
            <div className="veredicto">
              <div>
                <h2 className="ticket-titulo">{resultado.nombre}</h2>
                <p className="ticket-texto">{resultado.detalle}</p>
                <p className="confianza">Probabilidad estimada: {porcentaje(resultado.confianza)}</p>
              </div>
            </div>

            <div className="ticket-corte" />

            <h3 className="ticket-subtitulo">Probabilidad de cada resultado</h3>
            <ul className="barras">
              {resultado.probabilidades.map((p) => (
                <li key={p.clase} className={p.clase === resultado.clase ? "elegida" : undefined}>
                  <div className="barra-texto">
                    <span>
                      <strong>{p.nombre}</strong>
                    </span>
                    <span className="barra-valor">{porcentaje(p.probabilidad)}</span>
                  </div>
                  <div className="barra-pista" aria-hidden="true">
                    <div className="barra-relleno" style={{ width: `${Math.max(p.probabilidad * 100, 1)}%` }} />
                  </div>
                </li>
              ))}
            </ul>

            <div className="avisos">
              <h3 className="ticket-subtitulo">A tener en cuenta</h3>
              <ul>
                <li>Se predijo con datos nuevos.</li>
              </ul>
            </div>

          </>
        )}
      </div>
    </div>
  );
}
