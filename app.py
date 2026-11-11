
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path

# ==========================================================
# 1. CONFIGURACIÓN GENERAL
# ==========================================================

st.set_page_config(
    page_title="Ecuador Growth ML · MLOps",
    page_icon="📈",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent

# Las variables se recuperan del modelo entrenado y su configuración.
# No se definen manualmente en la aplicación.

st.title("📈 Ecuador Growth ML")
st.caption(
    "Predicción económica · Machine Learning · MLOps"
)

st.info(
    "Proyecto académico con datos simulados. "
    "Las estimaciones no son pronósticos oficiales."
)


# ==========================================================
# 2. FUNCIONES PARA LOCALIZAR ARCHIVOS
# ==========================================================

def localizar_archivo(nombre, carpetas):
    for carpeta in carpetas:
        ruta = ROOT / carpeta / nombre

        if ruta.exists():
            return ruta

    return None


@st.cache_resource
def cargar_modelo(ruta):
    return joblib.load(ruta)


@st.cache_data
def cargar_csv(ruta):
    return pd.read_csv(ruta)


@st.cache_data
def cargar_json(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def preparar_variables(datos):
    faltantes = [variable for variable in FEATURES if variable not in datos.columns]
    if faltantes:
        raise ValueError("Faltan variables requeridas: " + ", ".join(faltantes))

    X = datos.loc[:, FEATURES].copy()
    for variable in FEATURES:
        X[variable] = pd.to_numeric(X[variable], errors="coerce")

    if X.isna().all(axis=1).any():
        raise ValueError("Hay filas sin ningún predictor numérico válido.")
    # Las imputaciones parciales, si las hay, se gestionan en el pipeline entrenado.
    if X.isna().any().any() and not hasattr(modelo, "named_steps"):
        raise ValueError("Hay valores faltantes. Revise su CSV o el preprocesamiento del modelo.")
    return X


def nombres_modelo_entrenado(modelo):
    """Recupera los nombres de entrada cuando scikit-learn los conserva."""
    if hasattr(modelo, "feature_names_in_"):
        return list(modelo.feature_names_in_)
    if hasattr(modelo, "named_steps"):
        for paso in modelo.named_steps.values():
            if hasattr(paso, "feature_names_in_"):
                return list(paso.feature_names_in_)
    return None


# ==========================================================
# 3. CARGAR MODELO ENTRENADO
# ==========================================================

ruta_modelo = localizar_archivo(
    "modelo_pib.joblib",
    ["modelos", "."]
)

if ruta_modelo is None:
    st.error(
        "No se encontró modelo_pib.joblib. "
        "Colócalo en la carpeta modelos/ "
        "o en la raíz del repositorio."
    )
    st.stop()

try:
    modelo = cargar_modelo(str(ruta_modelo))
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()


# ==========================================================
# 4. CARGAR EXPERIMENTOS Y MÉTRICAS
# ==========================================================

CARPETAS_TRACKER = [
    "salidas_tracker",
    "datos",
    "."
]


def leer_tracker(nombre):
    ruta = localizar_archivo(
        nombre,
        CARPETAS_TRACKER
    )

    if ruta is None:
        return None

    return cargar_csv(str(ruta))


experimentos = leer_tracker("experimentos.csv")

folds = leer_tracker("experimentos_folds.csv")

predicciones_test = leer_tracker(
    "predicciones_test_ganador.csv"
)

ruta_mejor = localizar_archivo(
    "mejor_experimento.json",
    CARPETAS_TRACKER
)

mejor_experimento = (
    cargar_json(str(ruta_mejor))
    if ruta_mejor is not None
    else None
)


# ==========================================================
# 4B. CONFIGURACIÓN DINÁMICA DE VARIABLES
# ==========================================================

ruta_config = localizar_archivo(
    "config_modelo.json", ["salidas_tracker", "modelos", "."]
)
config = cargar_json(str(ruta_config)) if ruta_config else {}
nombres_entrenados = nombres_modelo_entrenado(modelo)

# Compatibilidad con el notebook anterior: mejor_experimento.json
# también contiene las variables elegidas en entrenamiento.
features_config = config.get("features")
if features_config is None and mejor_experimento is not None:
    features_config = mejor_experimento.get("features")
if features_config is None:
    features_config = nombres_entrenados

if not isinstance(features_config, list) or not features_config or not all(
    isinstance(v, str) and v.strip() for v in features_config
):
    st.error(
        "No se conoce el listado de variables del modelo. "
        "Suba config_modelo.json generado por Colab."
    )
    st.stop()

FEATURES = list(features_config)
TARGET = config.get("target", "crecimiento_pib")
if len(set(FEATURES)) != len(FEATURES):
    st.error("La configuración contiene variables duplicadas.")
    st.stop()

if nombres_entrenados is not None and nombres_entrenados != FEATURES:
    st.error(
        "Las variables de config_modelo.json no coinciden con las "
        "variables y el orden del modelo_pib.joblib. "
        "Vuelva a exportar ambos archivos desde la misma ejecución de Colab."
    )
    st.stop()

if nombres_entrenados is None:
    st.warning(
        "No fue posible contrastar los nombres de entrada guardados en el modelo. "
        "Verifique que la configuración y el .joblib sean de la misma ejecución."
    )

with st.expander(f"Variables del modelo activo ({len(FEATURES)})"):
    st.write(FEATURES)


# ==========================================================
# 5. ESTADO DE LA SESIÓN
# ==========================================================

if "historial" not in st.session_state:
    st.session_state.historial = []

if "prediccion_manual" not in st.session_state:
    st.session_state.prediccion_manual = None

if "registro_manual" not in st.session_state:
    st.session_state.registro_manual = None


# ==========================================================
# 6. MENÚ PRINCIPAL
# ==========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🧮 Predicción manual",
    "📂 Cargar CSV",
    "🧪 Experiment Tracker",
    "📊 Monitoreo"
])


# ==========================================================
# 7. PREDICCIÓN MANUAL
# ==========================================================

with tab1:

    st.header("Predicción manual del PIB")

    st.write(
        "Ingrese los valores de los indicadores "
        "económicos utilizados por el modelo."
    )

    with st.form("formulario_manual"):
        valores = {}
        columnas = st.columns(2)
        for i, variable in enumerate(FEATURES):
            with columnas[i % 2]:
                valores[variable] = st.number_input(
                    variable.replace("_", " ").capitalize(),
                    value=0.0,
                    format="%.4f",
                    key=f"predictor_{variable}"
                )

        trimestre = st.text_input("Trimestre de estimación", value="2026Q4")
        escenario = st.selectbox(
            "Escenario", ["Base", "Optimista", "Pesimista", "Personalizado"]
        )
        calcular = st.form_submit_button("Generar predicción", type="primary")

    if calcular:
        datos_manual = pd.DataFrame([valores], columns=FEATURES)
        try:
            resultado = float(modelo.predict(datos_manual)[0])
            st.session_state.prediccion_manual = resultado
            st.session_state.registro_manual = {
                "trimestre": trimestre,
                "escenario": escenario,
                "PIB_estimado": resultado,
                "n_variables": len(FEATURES)
            }
        except Exception as e:
            st.session_state.prediccion_manual = None
            st.session_state.registro_manual = None
            st.error(f"Error al predecir: {e}")

    if st.session_state.prediccion_manual is not None:

        st.metric(
            "Crecimiento estimado del PIB",
            f"{st.session_state.prediccion_manual:.2f}%"
        )

        if st.button(
            "Registrar predicción manual",
            key="registrar_manual"
        ):

            st.session_state.historial.append(
                st.session_state.registro_manual.copy()
            )

            st.success("Predicción registrada.")


# ==========================================================
# 8. CARGAR Y PROCESAR NUEVOS INDICADORES
# ==========================================================

with tab2:

    st.header("Predicción mediante archivo CSV")

    st.write(
        "Cargue un archivo CSV que contenga "
        f"las {len(FEATURES)} variables del modelo."
    )

    with st.expander("Ver variables requeridas"):

        st.dataframe(
            pd.DataFrame({
                "Variable": FEATURES
            }),
            hide_index=True
        )

    archivo = st.file_uploader(
        "Seleccione archivo CSV",
        type=["csv"],
        key="archivo_prediccion"
    )

    if archivo is not None:

        try:

            nuevos_datos = pd.read_csv(archivo)

            st.subheader("Información recibida")

            st.dataframe(
                nuevos_datos,
                use_container_width=True
            )

            X_nuevo = preparar_variables(
                nuevos_datos
            )

            predicciones = modelo.predict(X_nuevo)

            resultados = nuevos_datos.copy()

            resultados["PIB_estimado"] = predicciones

            st.success(
                f"Se generaron {len(predicciones)} "
                "predicciones."
            )

            st.subheader("Resultados")

            st.dataframe(
                resultados,
                use_container_width=True
            )

            st.download_button(
                "Descargar predicciones",
                data=resultados.to_csv(index=False),
                file_name="predicciones_pib.csv",
                mime="text/csv"
            )

            if st.button(
                "Registrar estimaciones",
                key="registrar_csv"
            ):

                registro = pd.DataFrame({
                    "trimestre": (
                        nuevos_datos["trimestre"]
                        if "trimestre" in nuevos_datos.columns
                        else "No especificado"
                    ),
                    "escenario": (
                        nuevos_datos["escenario"]
                        if "escenario" in nuevos_datos.columns
                        else "Sin escenario"
                    ),
                    "PIB_estimado": predicciones
                })

                st.session_state.historial.extend(
                    registro.to_dict("records")
                )

                st.success("Estimaciones registradas.")

        except Exception as e:

            st.error(
                f"No se pudo procesar el archivo: {e}"
            )

    # ------------------------------------------------------
    # HISTORIAL
    # ------------------------------------------------------

    st.divider()

    st.subheader("Historial de predicciones")

    if st.session_state.historial:

        historial = pd.DataFrame(
            st.session_state.historial
        )

        st.dataframe(
            historial,
            use_container_width=True
        )

        st.download_button(
            "Descargar historial",
            data=historial.to_csv(index=False),
            file_name="historial_predicciones.csv",
            mime="text/csv"
        )

        if st.button("Limpiar historial"):

            st.session_state.historial = []
            st.rerun()

    else:

        st.info(
            "Todavía no existen predicciones registradas."
        )


# ==========================================================
# 9. EXPERIMENT TRACKER
# ==========================================================

with tab3:

    st.header("ML Experiment Tracker")

    st.caption(
        "Seguimiento y comparación de experimentos "
        "de Machine Learning."
    )

    if experimentos is None:

        st.warning(
            "No se encontró experimentos.csv. "
            "Sube el archivo generado en Colab "
            "a la carpeta salidas_tracker/."
        )

    else:

        requeridas = [
            "run_id",
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "mae_cv",
            "rmse_cv"
        ]

        faltantes = [
            c for c in requeridas
            if c not in experimentos.columns
        ]

        if faltantes:

            st.error(
                "El archivo de experimentos no tiene "
                "las columnas: " + ", ".join(faltantes)
            )

        else:

            exp = experimentos.copy()

            columnas_numericas = [
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
                "mae_cv",
                "rmse_cv"
            ]

            for c in columnas_numericas:
                exp[c] = pd.to_numeric(
                    exp[c],
                    errors="coerce"
                )

            exp = exp.dropna(
                subset=["rmse_cv", "mae_cv"]
            )

            exp = exp.sort_values(
                "rmse_cv",
                ascending=True
            )

            if exp.empty:

                st.warning(
                    "No existen experimentos válidos."
                )

            else:

                ganador = exp.iloc[0]

                # ------------------------------------------
                # INDICADORES GENERALES
                # ------------------------------------------

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Ejecuciones",
                    len(exp)
                )

                col2.metric(
                    "Mejor RMSE CV",
                    f"{ganador['rmse_cv']:.4f}"
                )

                col3.metric(
                    "Mejor MAE CV",
                    f"{exp['mae_cv'].min():.4f}"
                )

                col4.metric(
                    "Run ganador",
                    str(ganador["run_id"])
                )

                st.divider()

                # ------------------------------------------
                # GRILLA DE HIPERPARÁMETROS
                # ------------------------------------------

                st.subheader(
                    "Grilla de hiperparámetros"
                )

                mostrar = [
                    "run_id",
                    "n_estimators",
                    "max_depth",
                    "min_samples_leaf",
                    "mae_cv",
                    "rmse_cv"
                ]

                if "modelo" in exp.columns:
                    mostrar.insert(1, "modelo")

                st.dataframe(
                    exp[mostrar],
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "Descargar experimentos",
                    data=exp.to_csv(index=False),
                    file_name="experimentos.csv",
                    mime="text/csv"
                )

                st.divider()

                # ------------------------------------------
                # SELECCIÓN DE EJECUCIONES
                # ------------------------------------------

                st.subheader(
                    "Comparación de modelos"
                )

                opciones = exp["run_id"].astype(
                    str
                ).tolist()

                seleccion = st.multiselect(
                    "Seleccione ejecuciones para comparar",
                    options=opciones,
                    default=opciones[:min(4, len(opciones))]
                )

                comparacion = exp[
                    exp["run_id"].astype(str).isin(
                        seleccion
                    )
                ].copy()

                if not comparacion.empty:

                    metrica = st.radio(
                        "Métrica de comparación",
                        ["rmse_cv", "mae_cv"],
                        horizontal=True,
                        format_func=lambda x: {
                            "rmse_cv": "RMSE",
                            "mae_cv": "MAE"
                        }[x]
                    )

                    grafico = comparacion[
                        ["run_id", metrica]
                    ].set_index("run_id")

                    st.bar_chart(grafico)

                    st.caption(
                        "Menor error indica mejor "
                        "desempeño predictivo."
                    )

                    st.subheader(
                        "Hiperparámetros comparados"
                    )

                    st.dataframe(
                        comparacion[mostrar],
                        use_container_width=True,
                        hide_index=True
                    )

                st.divider()

                # ------------------------------------------
                # DETALLE DE UNA EJECUCIÓN
                # ------------------------------------------

                st.subheader("Detalle de ejecución")

                run_seleccionado = st.selectbox(
                    "Seleccione un Run ID",
                    options=opciones
                )

                detalle = exp[
                    exp["run_id"].astype(str)
                    == run_seleccionado
                ].iloc[0]

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Número de árboles",
                    int(detalle["n_estimators"])
                )

                c2.metric(
                    "Profundidad máxima",
                    int(detalle["max_depth"])
                )

                c3.metric(
                    "Mínimo por hoja",
                    int(detalle["min_samples_leaf"])
                )

                c4, c5 = st.columns(2)

                c4.metric(
                    "MAE validación cruzada",
                    f"{detalle['mae_cv']:.4f}"
                )

                c5.metric(
                    "RMSE validación cruzada",
                    f"{detalle['rmse_cv']:.4f}"
                )

                # ------------------------------------------
                # RESULTADOS POR FOLD
                # ------------------------------------------

                st.divider()

                st.subheader(
                    "Validación cruzada temporal"
                )

                if folds is not None and "run_id" in folds.columns:

                    folds_run = folds[
                        folds["run_id"].astype(str)
                        == run_seleccionado
                    ].copy()

                    if not folds_run.empty:

                        st.dataframe(
                            folds_run,
                            use_container_width=True,
                            hide_index=True
                        )

                        if "rmse" in folds_run.columns:

                            grafico_folds = folds_run[
                                ["rmse"]
                            ].reset_index(drop=True)

                            grafico_folds.index += 1

                            grafico_folds.index.name = "Fold"

                            st.line_chart(grafico_folds)

                    else:

                        st.info(
                            "No hay folds registrados "
                            "para esta ejecución."
                        )

                else:

                    st.info(
                        "No se encontró "
                        "experimentos_folds.csv."
                    )

                # ------------------------------------------
                # MODELO SELECCIONADO
                # ------------------------------------------

                st.divider()

                st.subheader(
                    "Mejor experimento"
                )

                if mejor_experimento is not None:

                    st.success(
                        "Experimento seleccionado: "
                        + str(
                            mejor_experimento.get(
                                "run_id", "No disponible"
                            )
                        )
                    )

                    st.json(mejor_experimento)

                else:

                    st.info(
                        "No se encontró "
                        "mejor_experimento.json."
                    )


# ==========================================================
# 10. MONITOREO DEL MODELO
# ==========================================================

with tab4:

    st.header("Monitoreo del modelo")

    st.write(
        "Evaluación del comportamiento predictivo "
        "a partir de observaciones conocidas."
    )

    if predicciones_test is not None:

        st.subheader(
            "Predicciones del conjunto de prueba"
        )

        st.dataframe(
            predicciones_test,
            use_container_width=True
        )

        columnas = predicciones_test.columns.tolist()

        st.write(
            "Variables disponibles:",
            columnas
        )

        # Identificamos las columnas del notebook
        posibles_observados = [
            "crecimiento_pib",
            "PIB_real",
            "observado",
            "y_true"
        ]

        posibles_predichos = [
            "prediccion",
            "PIB_estimado",
            "predicho",
            "y_pred"
        ]

        col_real = next(
            (
                c for c in posibles_observados
                if c in columnas
            ),
            None
        )

        col_pred = next(
            (
                c for c in posibles_predichos
                if c in columnas
            ),
            None
        )

        if col_real and col_pred:

            evaluacion = predicciones_test.copy()

            evaluacion[col_real] = pd.to_numeric(
                evaluacion[col_real],
                errors="coerce"
            )

            evaluacion[col_pred] = pd.to_numeric(
                evaluacion[col_pred],
                errors="coerce"
            )

            evaluacion = evaluacion.dropna(
                subset=[col_real, col_pred]
            )

            if not evaluacion.empty:

                evaluacion["error"] = (
                    evaluacion[col_real]
                    - evaluacion[col_pred]
                )

                evaluacion["error_absoluto"] = (
                    evaluacion["error"].abs()
                )

                mae = evaluacion[
                    "error_absoluto"
                ].mean()

                rmse = np.sqrt(
                    np.mean(
                        evaluacion["error"] ** 2
                    )
                )

                c1, c2 = st.columns(2)

                c1.metric(
                    "MAE de prueba",
                    f"{mae:.4f}"
                )

                c2.metric(
                    "RMSE de prueba",
                    f"{rmse:.4f}"
                )

                grafico = evaluacion[
                    [col_real, col_pred]
                ].copy()

                if "trimestre" in evaluacion.columns:

                    grafico.index = (
                        evaluacion["trimestre"]
                        .astype(str)
                    )

                st.subheader(
                    "Observado vs. predicho"
                )

                st.line_chart(grafico)

                st.subheader(
                    "Errores de predicción"
                )

                errores = evaluacion[
                    ["error", "error_absoluto"]
                ].copy()

                if "trimestre" in evaluacion.columns:

                    errores.index = (
                        evaluacion["trimestre"]
                        .astype(str)
                    )

                st.bar_chart(errores)

            else:

                st.warning(
                    "No existen observaciones válidas "
                    "para calcular las métricas."
                )

        else:

            st.warning(
                "No se identificaron automáticamente "
                "las columnas observada y predicha."
            )

    else:

        st.info(
            "Para visualizar métricas de prueba, "
            "incorpore predicciones_test_ganador.csv "
            "a salidas_tracker/."
        )

    # ------------------------------------------------------
    # HISTORIAL DE ESTIMACIONES DE LA SESIÓN
    # ------------------------------------------------------

    st.divider()

    st.subheader(
        "Predicciones registradas durante la sesión"
    )

    if st.session_state.historial:

        historial = pd.DataFrame(
            st.session_state.historial
        )

        st.dataframe(
            historial,
            use_container_width=True
        )

        st.download_button(
            "Descargar historial de monitoreo",
            data=historial.to_csv(index=False),
            file_name="historial_monitoreo.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "Aún no se han registrado estimaciones "
            "durante esta sesión."
        )

    st.caption(
        "El historial de sesión no representa "
        "observaciones reales ni permite calcular "
        "errores hasta disponer del PIB observado."
    )


# ==========================================================
# 11. PIE DE PÁGINA
# ==========================================================

st.divider()

st.caption(
    "Ecuador Growth ML · Proyecto académico de MLOps "
    "| Google Colab + GitHub + Streamlit"
)
