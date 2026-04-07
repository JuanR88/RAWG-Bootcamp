
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from dotenv import load_dotenv
import nbformat
import papermill as pm
from pydantic import BaseModel

import joblib


app = FastAPI(title="RAWG Notebooks Runner", version="0.2.0")


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
NOTEBOOK_DIR = PROJECT_ROOT / "NoteBooks" / "Untitled Folder"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

NOTEBOOKS = {
    "entrenamiento": NOTEBOOK_DIR / "Entrenamiento.ipynb",
    "visualizacion": NOTEBOOK_DIR / "Visualizacion.ipynb",
    "preguntas": NOTEBOOK_DIR / "Preguntas.ipynb",
}


def _load_success_artifacts() -> tuple[object, list[str], dict]:
    model_path = ARTIFACTS_DIR / "success_model.joblib"
    features_path = ARTIFACTS_DIR / "success_features.joblib"
    stats_path = ARTIFACTS_DIR / "success_stats.joblib"

    if not model_path.exists() or not features_path.exists() or not stats_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Faltan artefactos del modelo. Ejecuta Entrenamiento.ipynb y guarda: "
                f"{model_path.name}, {features_path.name}, {stats_path.name} en {os.fspath(ARTIFACTS_DIR)}"
            ),
        )

    model = joblib.load(model_path)
    features = joblib.load(features_path)
    stats = joblib.load(stats_path)

    if not isinstance(features, list) or not all(isinstance(x, str) for x in features):
        raise HTTPException(status_code=500, detail="success_features.joblib inválido")
    if not isinstance(stats, dict):
        raise HTTPException(status_code=500, detail="success_stats.joblib inválido")

    return model, features, stats


def _run_notebook(
    notebook_path: Path,
    *,
    timeout_seconds: int = 1800,
    kernel_name: str | None = None,
    parameters: dict[str, object] | None = None,
) -> dict:
    if not notebook_path.exists():
        raise HTTPException(status_code=404, detail=f"No existe el notebook: {notebook_path}")

    output_dir = notebook_path.parent / "_executed"
    output_dir.mkdir(parents=True, exist_ok=True)

    executed_path = output_dir / notebook_path.name

    try:
        pm.execute_notebook(
            input_path=os.fspath(notebook_path),
            output_path=os.fspath(executed_path),
            parameters=parameters or {},
            kernel_name=kernel_name,
            cwd=os.fspath(notebook_path.parent),
            request_save_on_cell_execute=False,
            execution_timeout=int(timeout_seconds),
        )
        ok = True
        err = ""
    except Exception as e:
        ok = False
        err = str(e)

    respuesta_texto = ""
    images_png_base64: list[str] = []

    if executed_path.exists():
        nb = nbformat.read(executed_path, as_version=4)
        for cell in nb.cells:
            if cell.get("cell_type") != "code":
                continue
            for output in cell.get("outputs", []):
                output_type = output.get("output_type")

                if output_type == "stream":
                    respuesta_texto += output.get("text", "")

                elif output_type == "execute_result":
                    data = output.get("data", {})
                    if "text/plain" in data:
                        respuesta_texto += str(data["text/plain"]) + "\n"

                elif output_type == "display_data":
                    data = output.get("data", {})
                    if "text/plain" in data:
                        respuesta_texto += str(data["text/plain"]) + "\n"
                    if "image/png" in data:
                        images_png_base64.append(data["image/png"])

                elif output_type == "error":
                    traceback = output.get("traceback", [])
                    if traceback:
                        respuesta_texto += "\n".join(traceback) + "\n"

    return {
        "ok": ok,
        "returncode": 0 if ok else 1,
        "notebook": os.fspath(notebook_path),
        "parameters": parameters or {},
        "executed_notebook": os.fspath(executed_path) if executed_path.exists() else None,
        "respuesta": respuesta_texto,
        "images_png_base64": images_png_base64,
        "plot_count": len(images_png_base64),
        "stdout": "",
        "stderr": err,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "Conectado"}



class SuccessPredictRequest(BaseModel):
    genres: list[str]


@app.post("/predict/success")
def predict_success(body: SuccessPredictRequest) -> dict:
    model, features, stats = _load_success_artifacts()

    import numpy as np
    import pandas as pd

    x = pd.DataFrame(np.zeros((1, len(features))), columns=features)

    for genre in body.genres:
        if not isinstance(genre, str):
            continue
        col = f"genre_{genre.lower().replace(' ', '_')}"
        if col in x.columns:
            x[col] = 1

    if "reviews_text_count" in x.columns:
        x["reviews_text_count"] = float(stats.get("reviews_text_count_q75", 0.0))
    if "metacritic" in x.columns:
        x["metacritic"] = float(stats.get("metacritic_default", 0.0))
    if "added" in x.columns:
        x["added"] = float(stats.get("added_q75", 0.0))
    if "suggestions_count" in x.columns:
        x["suggestions_count"] = float(stats.get("suggestions_count_q75", 0.0))
    if "release_year" in x.columns:
        x["release_year"] = float(stats.get("release_year_mean", 0.0))
    if "release_month" in x.columns:
        x["release_month"] = float(stats.get("release_month_default", 6.0))
    if "metacritic_missing" in x.columns:
        x["metacritic_missing"] = 0

    try:
        prob = float(model.predict_proba(x)[0][1])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Error prediciendo: {e}")

    threshold = float(stats.get("threshold", 0.4))
    pred = int(prob > threshold)

    return {
        "prediction": pred,
        "probability": prob,
        "threshold": threshold,
        "genres": body.genres,
    }


@app.post("/run/entrenamiento")
def run_entrenamiento(
    timeout_seconds: int = Query(default=1800, ge=1, le=24 * 3600),
    kernel_name: str | None = Query(default=None),
) -> dict:
    result = _run_notebook(
        NOTEBOOKS["entrenamiento"],
        timeout_seconds=timeout_seconds,
        kernel_name=kernel_name,
    )
    if not result["ok"]:
        raise HTTPException(status_code=500, detail=result)
    return result


class VisualizacionRequest(BaseModel):
    question: str
    min_year: int = 2000
    max_year: int = 2020


@app.post("/run/visualizacion")
async def run_visualizacion(
    request: Request,
    body: VisualizacionRequest | None = None,
    question: str | None = Query(default=None),
    min_year: int | None = Query(default=None, ge=1980, le=2100),
    max_year: int | None = Query(default=None, ge=1980, le=2100),
    timeout_seconds: int = Query(default=1800, ge=1, le=24 * 3600),
    kernel_name: str | None = Query(default=None),
) -> dict:
    q: str | None = None
    y_min: int = 2000
    y_max: int = 2020

    if body and body.question is not None:
        q = body.question
        y_min = body.min_year
        y_max = body.max_year
    elif question is not None:
        q = question
        if min_year is not None:
            y_min = min_year
        if max_year is not None:
            y_max = max_year
    else:
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            payload = None

        if isinstance(payload, dict):
            raw_q = payload.get("question")
            if isinstance(raw_q, str):
                q = raw_q
            raw_min = payload.get("min_year")
            raw_max = payload.get("max_year")
            if isinstance(raw_min, int):
                y_min = raw_min
            if isinstance(raw_max, int):
                y_max = raw_max

    if q is None or not q.strip():
        raise HTTPException(
            status_code=422,
            detail="Falta 'question'. Envia JSON {'question': '...','min_year':2000,'max_year':2020}",
        )

    if y_min > y_max:
        raise HTTPException(status_code=422, detail="min_year no puede ser mayor que max_year")

    params = {
        "question": q.strip(),
        "min_year": int(y_min),
        "max_year": int(y_max),
        "run_api": True,
    }

    result = _run_notebook(
        NOTEBOOKS["visualizacion"],
        timeout_seconds=timeout_seconds,
        kernel_name=kernel_name,
        parameters=params,
    )

    if not result["ok"]:
        raise HTTPException(status_code=500, detail=result)

    return result


class PreguntaRequest(BaseModel):
    question: str


@app.post("/run/preguntas")
async def run_preguntas(
    request: Request,
    body: PreguntaRequest | None = None,
    question: str | None = Query(default=None),
    timeout_seconds: int = Query(default=1800, ge=1, le=24 * 3600),
    kernel_name: str | None = Query(default=None),
) -> dict:
    q: str | None = None
    if body and body.question is not None:
        q = body.question
    elif question is not None:
        q = question
    else:
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            payload = None

        if isinstance(payload, dict):
            raw_q = payload.get("question")
            if isinstance(raw_q, str):
                q = raw_q

    if q is None or not q.strip():
        raise HTTPException(
            status_code=422,
            detail="Falta 'question'. Envia JSON {'question': '...'} o usa ?question=...",
        )

    params = {"question": q}

    result = _run_notebook(
        NOTEBOOKS["preguntas"],
        timeout_seconds=timeout_seconds,
        kernel_name=kernel_name,
        parameters=params,
    )
    if not result["ok"]:
        raise HTTPException(status_code=500, detail=result)
    return result



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
