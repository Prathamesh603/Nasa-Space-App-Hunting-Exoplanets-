# server.py (MULTI-MODEL SUPPORT - UPDATED FOR 'web' FOLDER)
import os, io, uuid, json, random, traceback, logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np
import joblib

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("exoplanet-api")

def raise_500(e: Exception):
    tb = traceback.format_exc()
    log.error(tb)
    raise HTTPException(status_code=500, detail=tb)

# ---------- Config ----------
BASE_DIR = Path(__file__).parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))
PRED_DIR = BASE_DIR / "predictions"
PRED_DIR.mkdir(exist_ok=True, parents=True)

# *** IMPORTANT CHANGE: Reference the 'web' folder ***
WEB_DIR = BASE_DIR / "web" 
WEB_DIR.mkdir(exist_ok=True, parents=True) # Ensure 'web' folder exists

MAX_SAMPLE = 2000
TOPN_CORR = 15
TOPK_FIMP = 30
ROC_PR_MAX_PTS = 200

# Model Mapping: KEY -> Filename
MODEL_MAP = {
    "koi": MODELS_DIR / "model_koi.pkl",
    "k2": MODELS_DIR / "model_k2.pkl",
    "tess": MODELS_DIR / "model_TESS.pkl",
}
VALID_MODEL_KEYS = list(MODEL_MAP.keys())

# ---------- App ----------
app = FastAPI(title="Exoplanet Classifier API (Multi-Model)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# *** IMPORTANT CHANGE: Mount the 'web' folder instead of 'static' ***
# This makes files inside 'web' accessible at the /web/ URL prefix (e.g., /web/images/logo.png)
app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

# ---------- Model Loading and Feature Extraction ----------
LOADED_MODELS: Dict[str, Any] = {}
MODEL_FEATURES: Dict[str, List[str]] = {}

def load_model(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Model not found at {p}")
    return joblib.load(p)

def get_train_features(m) -> List[str]:
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)
    try:
        # Assumes the final estimator in a pipeline has the attribute
        return list(getattr(m[-1], "feature_names_in_", []))
    except Exception:
        return []

# Load all models at startup
for key, path in MODEL_MAP.items():
    try:
        model = load_model(path)
        LOADED_MODELS[key] = model
        MODEL_FEATURES[key] = get_train_features(model)
        log.info(f"Loaded model '{key}' from {path.name} with {len(MODEL_FEATURES[key])} features.")
    except Exception as e:
        log.error(f"Failed to load model {key} from {path.name}: %s", e)
        # Re-raise to prevent server startup if models are essential
        raise

if not LOADED_MODELS:
    raise RuntimeError("No models were successfully loaded. Cannot start API.")


def get_feature_importances(model, feature_names):
    if hasattr(model, "feature_importances_"):
        s = pd.Series(model.feature_importances_, index=feature_names)
        return (
            s.reset_index(name="importance")
             .rename(columns={"index": "feature"})
             .sort_values("importance", ascending=False)
        )
    try:
        final_est = model[-1] if hasattr(model, "__getitem__") else model
        if hasattr(final_est, "feature_importances_"):
            s = pd.Series(final_est.feature_importances_, index=feature_names)
            return (
                s.reset_index(name="importance")
                 .rename(columns={"index": "feature"})
                 .sort_values("importance", ascending=False)
            )
    except Exception:
        pass
    return pd.DataFrame({"feature": feature_names, "importance": np.nan})

# ---------- Labels ----------
CANON_LABELS = ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"]
LABEL_TO_CODE = {lab: i for i, lab in enumerate(CANON_LABELS)}
CODE_TO_LABEL = {i: lab for lab, i in LABEL_TO_CODE.items()}

def normalize_label_str(s: str) -> str:
    up = s.strip().upper().replace("_", " ").replace("-", " ")
    if up in {"FALSE POSITIVE", "FALSEPOSITIVE"}: return "FALSE POSITIVE"
    if up == "CONFIRMED": return "CONFIRMED"
    if up in {"CANDIDATE", "CANDIDATES"}: return "CANDIDATE"
    return s  # return original if unknown

def gt_series_to_canonical(series: pd.Series) -> pd.Series:
    """Map GT values to canonical labels; leave invalid as NaN (no crashes)."""
    def map_one(x):
        if pd.isna(x):
            return np.nan
        try:
            xi = int(x)
            return CODE_TO_LABEL.get(xi, np.nan)
        except Exception:
            pass
        try:
            s = str(x)
        except Exception:
            return np.nan
        lab = normalize_label_str(s)
        return lab if lab in CANON_LABELS else np.nan
    return series.map(map_one)

# ---------- Utils ----------
def json_safe_preview(df: pd.DataFrame, n=100):
    return json.loads(df.head(n).to_json(orient="records"))

def corr_topn(df_num: pd.DataFrame, topn: int) -> Dict[str, Any]:
    if df_num.shape[1] == 0:
        return {"labels": [], "matrix": []}
    variances = df_num.var().sort_values(ascending=False)
    cols = list(variances.index[:min(topn, len(variances))])
    sub = df_num[cols]
    c = sub.corr(method="pearson").fillna(0.0)
    return {"labels": list(c.columns), "matrix": c.values.tolist()}

def downsample_indices(n: int, k: int) -> np.ndarray:
    if n <= k: return np.arange(n)
    return np.array(random.sample(range(n), k))

# ---------- Routes ----------
@app.get("/", include_in_schema=False)
def root():
    # Still redirects to the /predict.html route
    return RedirectResponse("/predict.html")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

@app.get("/predict.html", response_class=HTMLResponse)
def predict_page():
    # *** IMPORTANT CHANGE: Look for predict.html inside the 'web' folder ***
    p = WEB_DIR / "predict.html"
    if p.exists():
        return HTMLResponse(content=p.read_text(encoding="utf-8"))
    raise HTTPException(status_code=500, detail="predict.html not found in the 'web' folder.")

@app.get("/api/model-keys")
def model_keys():
    """Returns the list of available model keys."""
    return {"model_keys": VALID_MODEL_KEYS}


@app.get("/api/required-features")
def required_features(model_key: str = Query(...)):
    """Returns required features for a specific model key."""
    if model_key not in MODEL_FEATURES:
        raise HTTPException(status_code=400, detail=f"Invalid model_key: {model_key}")
    return {"required_features": MODEL_FEATURES[model_key]}

@app.get("/api/model-info")
def model_info(model_key: str = Query(...)):
    """Returns detailed info for a specific model key."""
    model = LOADED_MODELS.get(model_key)
    if model is None:
        raise HTTPException(status_code=400, detail=f"Invalid model_key: {model_key}")
    
    train_features = MODEL_FEATURES.get(model_key, [])
    
    return {
        "model_type": type(model).__name__,
        "model_key": model_key,
        "n_train_features": len(train_features),
        "train_features": train_features,
    }


@app.post("/api/analyze-json")
async def analyze_json(file: UploadFile = File(...), model_key: str = Form(...)):
    try:
        model = LOADED_MODELS.get(model_key)
        if model is None:
            raise HTTPException(status_code=400, detail=f"Invalid model_key: {model_key}")
            
        TRAIN_FEATURES = MODEL_FEATURES[model_key]

        content = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(content))
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(content), encoding="latin1")

        # -------- NEW: make a numeric-only copy for EDA ----------
        df_num = df.apply(pd.to_numeric, errors="coerce")

        n_rows, n_cols = int(len(df)), int(df.shape[1])
        cols = list(df.columns)

        # Missing counts computed on original df (as users see it)
        miss = df.isna().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        missing_counts = {str(k): int(v) for k, v in miss.items()}

        # -------- use the numeric copy for EDA ----------
        num_cols = df_num.select_dtypes(include=[np.number]).columns.tolist()

        sample_vals: Dict[str, list] = {}
        if num_cols:
            idx = downsample_indices(n_rows, MAX_SAMPLE)
            df_sample = df_num.iloc[idx]
            for c in num_cols:
                vals = df_sample[c].dropna().astype(float).tolist()
                sample_vals[c] = vals

        corr = corr_topn(df_num.select_dtypes(include=[np.number]), TOPN_CORR)

        # Feature importance (if model supports it)
        fi = None
        if TRAIN_FEATURES:
            fi_df = get_feature_importances(model, TRAIN_FEATURES)
            if fi_df is not None:
                fi = fi_df.head(min(TOPK_FIMP, len(fi_df))).to_dict(orient="records")

        return JSONResponse({
            "ok": True,
            "n_rows": n_rows, "n_cols": n_cols,
            "columns_found": cols,
            "numeric_cols": num_cols,
            "missing_counts": missing_counts,
            "preview": json_safe_preview(df),
            "corr": corr,
            "sample_values": sample_vals,
            "feature_importance": fi
        })
    except Exception as e:
        raise_500(e)


@app.post("/api/predict-json")
async def predict_json(
    file: UploadFile = File(...),
    model_key: str = Form(...),
    gt_col: Optional[str] = Form(None)
):
    try:
        model = LOADED_MODELS.get(model_key)
        if model is None:
            raise HTTPException(status_code=400, detail=f"Invalid model_key: {model_key}")
            
        TRAIN_FEATURES = MODEL_FEATURES[model_key]

        content = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(content))
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(content), encoding="latin1")

        provided = set(df.columns)
        missing = [c for c in TRAIN_FEATURES if c not in provided]
        if missing:
            return JSONResponse({
                "ok": False,
                "message": f"Missing required columns for model '{model_key}': {len(missing)}",
                "missing": missing,
                "columns_found": list(df.columns),
                "n_rows": int(len(df))
            })

        X = df[TRAIN_FEATURES].copy()
        y_pred = model.predict(X)

        out = df.copy()
        # Ensure prediction output is a string label
        out["prediction"] = [normalize_label_str(str(p)) if str(p).upper() in {"CANDIDATE","CONFIRMED","FALSE POSITIVE"} else CODE_TO_LABEL.get(int(p), str(p)) if str(p).isdigit() else str(p) for p in y_pred]

        # Add confidence if available
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)
                out["pred_confidence"] = proba.max(axis=1)
            except Exception:
                proba = None

        preview = json_safe_preview(out)
        label_counts = {str(k): int(v) for k, v in out["prediction"].value_counts().items()}

        # Confidence histogram
        conf_hist = None
        if "pred_confidence" in out.columns:
            counts, bins = np.histogram(out["pred_confidence"].values, bins=20, range=(0.0, 1.0))
            conf_hist = {"bins": bins.tolist(), "counts": counts.astype(int).tolist()}

        # ---- Evaluation (only if GT looks valid) ----
        eval_block: Dict[str, Any] = {}
        if gt_col and gt_col in out.columns:
            y_true_all = gt_series_to_canonical(out[gt_col])
            y_pred_all = gt_series_to_canonical(out["prediction"])

            mask_ok = y_true_all.notna() & y_pred_all.notna()
            y_true = y_true_all[mask_ok]
            y_pred_ser = y_pred_all[mask_ok]

            if len(y_true) > 0:
                acc = float((y_true == y_pred_ser).mean())
                report = classification_report(
                    y_true, y_pred_ser, labels=CANON_LABELS, output_dict=True, zero_division=0
                )
                cm = confusion_matrix(y_true, y_pred_ser, labels=CANON_LABELS).tolist()
                y_true_code = y_true.map(LABEL_TO_CODE).astype(int).tolist()
                y_pred_code = y_pred_ser.map(LABEL_TO_CODE).astype(int).tolist()

                eval_block = {
                    "accuracy": acc,
                    "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
                    "report": report,
                    "confusion": {"labels": CANON_LABELS, "matrix": cm},
                    "line_series": {
                        "index": list(range(len(y_true))),
                        "actual": y_true_code,
                        "predicted": y_pred_code
                    },
                    "eval_rows": int(len(y_true)),
                    "dropped_rows": int(len(mask_ok) - len(y_true)),
                }

                # ROC / PR if we have probabilities
                if proba is not None:
                    try:
                        # Logic for aligning proba columns to CANON_LABELS order
                        model_classes = getattr(model, "classes_", None)
                        if model_classes is not None and len(model_classes) == len(CANON_LABELS):
                            mclasses = [normalize_label_str(str(c)) for c in model_classes]
                            # Create an index map to reorder proba columns
                            idx_map = [mclasses.index(c) for c in CANON_LABELS]
                            proba_aligned = proba[:, idx_map]
                        else:
                            # Assume proba is already in CANON_LABELS order or model classes aren't exposed
                            proba_aligned = proba

                        y_bin = label_binarize(y_true, classes=CANON_LABELS)
                        roc_curves, pr_curves = [], []
                        
                        # Only use proba for the rows that passed the GT check (mask_ok)
                        proba_valid_rows = proba_aligned[mask_ok]
                        
                        for i, lab in enumerate(CANON_LABELS):
                            # ROC Curve
                            fpr, tpr, _ = roc_curve(y_bin[:, i], proba_valid_rows[:, i])
                            step = max(1, len(fpr) // ROC_PR_MAX_PTS)
                            fpr, tpr = fpr[::step], tpr[::step]
                            roc_curves.append({"label": lab, "fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))})

                            # Precision-Recall Curve
                            precision, recall, _ = precision_recall_curve(y_bin[:, i], proba_valid_rows[:, i])
                            step = max(1, len(recall) // ROC_PR_MAX_PTS)
                            precision, recall = precision[::step], recall[::step]
                            pr_curves.append({"label": lab, "precision": precision.tolist(), "recall": recall.tolist()})
                        
                        if roc_curves: eval_block["roc"] = roc_curves
                        if pr_curves:  eval_block["pr"]  = pr_curves
                    except Exception as e:
                        log.error(f"Error generating ROC/PR for model {model_key}: {e}")
                        pass
            else:
                eval_block = {
                    "accuracy": 0.0, "macro_f1": 0.0,
                    "confusion": {"labels": CANON_LABELS, "matrix": [[0,0,0],[0,0,0],[0,0,0]]},
                    "line_series": {"index": [], "actual": [], "predicted": []},
                    "eval_rows": 0, "dropped_rows": int(len(out))
                }

        # Persist full CSV
        job_id = uuid.uuid4().hex
        (PRED_DIR / f"{job_id}.csv").write_text(out.to_csv(index=False, encoding="utf-8"))

        return JSONResponse({
            "ok": True,
            "n_rows": int(len(out)),
            "columns_found": list(df.columns),
            "missing": [],
            "preview": preview,
            "label_counts": label_counts,
            "has_confidence": bool("pred_confidence" in out.columns),
            "confidence_hist": conf_hist,
            "job_id": job_id,
            "eval": eval_block
        })
    except Exception as e:
        raise_500(e)

@app.get("/api/predict-scatter")
def predict_scatter(job_id: str = Query(...), x: str = Query(...), y: str = Query(...)):
    # Model key is not needed here as scatter plot uses the previously saved CSV
    try:
        path = PRED_DIR / f"{job_id}.csv"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Invalid job_id")
        df = pd.read_csv(path)
        for col in (x, y, "prediction"):
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found in predictions file")

        vals = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        vals["prediction"] = df.loc[vals.index, "prediction"]
        n = len(vals)
        if n == 0:
            return JSONResponse({"x": [], "y": [], "prediction": []})
        idx = downsample_indices(n, MAX_SAMPLE)
        sub = vals.iloc[idx]
        return JSONResponse({
            "x": sub[x].astype(float).tolist(),
            "y": sub[y].astype(float).tolist(),
            "prediction": sub["prediction"].astype(str).tolist()
        })
    except Exception as e:
        raise_500(e)

@app.get("/api/download/{job_id}")
def download(job_id: str):
    path = PRED_DIR / f"{job_id}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Invalid job_id")
    return FileResponse(str(path), media_type="text/csv", filename="predictions_with_labels.csv")

@app.get("/health")
def health():
    return {"ok": True, "models_loaded": list(LOADED_MODELS.keys())}