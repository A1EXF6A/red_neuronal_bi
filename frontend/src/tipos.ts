export type Tarea = "pedidos" | "resenas" | "precios";
export type TipoCampo = "numero" | "entero" | "texto" | "fecha" | "booleano";
export type ValorCampo = string | boolean;

export interface Condicion {
  campo: string;
  valor: boolean | string | number;
}

export interface CampoEsquema {
  nombre: string;
  etiqueta: string;
  tipo: TipoCampo;
  requerido: boolean;
  ejemplo?: string | number | boolean;
  defecto?: boolean;
  unidad?: string;
  ayuda?: string;
  min?: number;
  max?: number;
  mostrar_si?: Condicion;
  valores_validos?: string[];
  oculto?: boolean;
}

export interface SeccionEsquema {
  id: string;
  titulo: string;
  descripcion?: string;
  campos: CampoEsquema[];
}

export interface InfoClase {
  clase: string;
  nombre: string;
  detalle: string;
}

export interface MetricasTest {
  accuracy: number;
  macro_f1: number;
  macro_precision: number;
  macro_recall: number;
}

export interface Esquema {
  tarea: Tarea;
  nombre_corto: string;
  titulo: string;
  descripcion: string;
  momento_prediccion: string;
  secciones: SeccionEsquema[];
  clases: InfoClase[];
  tipo_modelo?: string | null;
  metricas_test?: MetricasTest | null;
  rango_fechas?: string[] | null;
}

export interface ProbabilidadClase extends InfoClase {
  probabilidad: number;
}

export interface VariableValor {
  nombre: string;
  valor: string | number;
}

export interface Prediccion {
  tarea: Tarea;
  clase: string;
  nombre: string;
  detalle: string;
  confianza: number;
  probabilidades: ProbabilidadClase[];
  variables_calculadas: VariableValor[];
  variables_imputadas: VariableValor[];
  advertencias: string[];
}

export interface ErrorCampo {
  campo: string;
  mensaje: string;
}

export interface EstadoRed {
  artefactos: boolean;
  modelo: boolean;
  error: string | null;
}

export interface Salud {
  estado: string;
  redes: Record<Tarea, EstadoRed>;
}
