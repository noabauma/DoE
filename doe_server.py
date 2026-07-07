"""Web UI for interactive DoE sessions (propose -> measure -> adapt).

Usage
-----
    .venv/bin/python doe_server.py [session.json] [--port 8080] [--host 127.0.0.1]

Serves the GUI on http://localhost:8080 -- forward it over ssh with

    ssh -L 8080:localhost:8080 <user>@<lab machine>

If the session file exists it is resumed, otherwise the browser shows the
setup wizard. Every result entered in the GUI is saved to the session file
immediately, and the same file also works with the interactive_doe.py CLI.
"""

import argparse
import csv
import io
import json
import math
import threading
from pathlib import Path

import torch
from flask import Flask, Response, jsonify, request, send_from_directory

from doe import IngredientCosts, InteractiveOptimizer

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(ROOT / "webui"), static_url_path="/static")

LOCK = threading.RLock()
STATE = {"opt": None, "path": "session.json", "pending": []}


class ApiError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


@app.errorhandler(ApiError)
def _api_error(err):
    return jsonify({"error": str(err)}), err.status


def _opt():
    if STATE["opt"] is None:
        raise ApiError("no session configured yet", 409)
    return STATE["opt"]


def _save():
    """Persist the session; pending proposals ride along in the same file."""
    STATE["opt"].save(STATE["path"])
    with open(STATE["path"]) as f:
        data = json.load(f)
    data["pending"] = STATE["pending"]
    with open(STATE["path"], "w") as f:
        json.dump(data, f, indent=2)


def _load(path):
    STATE["path"] = path
    if Path(path).exists():
        STATE["opt"] = InteractiveOptimizer.load(path)
        with open(path) as f:
            STATE["pending"] = json.load(f).get("pending", [])


def _floats(values, n, what):
    try:
        out = [float(v) for v in values]
    except (TypeError, ValueError):
        raise ApiError(f"{what}: expected numbers")
    if len(out) != n:
        raise ApiError(f"{what}: expected {n} values, got {len(out)}")
    if any(not math.isfinite(v) for v in out):
        raise ApiError(f"{what}: values must be finite")
    return out


def _point(opt, info):
    """Tensors in a best()/predicted_best() dict -> JSON-friendly."""
    if info is None:
        return None
    out = {"x": info["x"].tolist(), "y": info["y"].tolist()}
    for key in ("index", "cost", "cost_per_yield"):
        if key in info:
            out[key] = info[key]
    return out


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/state")
def api_state():
    with LOCK:
        opt = STATE["opt"]
        if opt is None:
            return jsonify({"configured": False, "session": STATE["path"]})
        resp = {
            "configured": True,
            "session": STATE["path"],
            "factor_names": opt.factor_names,
            "response_names": opt.response_names,
            "bounds": opt.bounds.tolist(),
            "target_task": opt.target_task,
            "maximize": opt.maximize,
            "cost_aware": opt.cost_aware,
            "num_init": opt.num_init,
            "num_results": opt.num_results,
            "results_x": opt.train_x.tolist(),
            "results_y": opt.train_y.tolist(),
            "pending": STATE["pending"],
            "best": _point(opt, opt.best()),
            "costs": None,
            "result_costs": [],
            "spend": 0.0,
            "projected": {},
        }
        if opt.costs is not None:
            resp["costs"] = {
                "prices": opt.costs.prices.tolist(),
                "fixed_cost": opt.costs.fixed_cost,
                "currency": opt.costs.currency,
            }
            if opt.num_results:
                resp["result_costs"] = opt.costs.experiment_cost(opt.train_x).tolist()
            resp["spend"] = opt.spend()
            resp["projected"] = {str(n): opt.projected_spend(n) for n in (5, 10, 20)}
        return jsonify(resp)


@app.post("/api/setup")
def api_setup():
    with LOCK:
        if STATE["opt"] is not None:
            raise ApiError("session already configured -- start the server "
                           "with a new session file for a fresh DoE", 409)
        cfg = request.get_json(force=True)
        factors = cfg.get("factors") or []
        responses = [str(r).strip() or f"response {i + 1}"
                     for i, r in enumerate(cfg.get("responses") or [])]
        if not factors:
            raise ApiError("add at least one ingredient")
        if not responses:
            raise ApiError("add at least one response")
        names, bounds, prices = [], [], []
        for i, f in enumerate(factors):
            name = str(f.get("name", "")).strip() or f"ingredient {i + 1}"
            lo, hi = _floats([f.get("low"), f.get("high")], 2, name)
            if not lo < hi:
                raise ApiError(f"{name}: min must be smaller than max")
            price, = _floats([f.get("price", 0)], 1, f"{name} price")
            names.append(name)
            bounds.append([lo, hi])
            prices.append(price)
        fixed, = _floats([cfg.get("fixed_cost", 0)], 1, "fixed cost")
        target = int(cfg.get("target_task", 0))
        if not 0 <= target < len(responses):
            raise ApiError("target response out of range")
        STATE["opt"] = InteractiveOptimizer(
            bounds=bounds,
            num_tasks=len(responses),
            target_task=target,
            maximize=bool(cfg.get("maximize", True)),
            costs=IngredientCosts(prices, fixed_cost=fixed, names=names,
                                  currency=str(cfg.get("currency", "USD")).strip() or "USD"),
            cost_aware=bool(cfg.get("cost_aware", False)),
            num_init=max(2, min(50, int(cfg.get("num_init", 4)))),
            factor_names=names,
            response_names=responses,
        )
        STATE["pending"] = []
        _save()
        return api_state()


@app.post("/api/suggest")
def api_suggest():
    with LOCK:
        opt = _opt()
        n = max(1, min(3, int(request.get_json(force=True).get("n", 1))))
        mode = "init" if opt.num_results < opt.num_init else "gp"
        proposals = opt.suggest(n)
        STATE["pending"] = [p.tolist() for p in proposals]
        _save()
        return jsonify({"mode": mode, "proposals": STATE["pending"]})


@app.post("/api/result")
def api_result():
    with LOCK:
        opt = _opt()
        data = request.get_json(force=True)
        x = _floats(data.get("x"), opt.num_factors, "concentrations")
        y = _floats(data.get("y"), opt.num_tasks, "responses")
        opt.add_result(x, y)
        idx = data.get("pending_index")
        if idx is not None and 0 <= int(idx) < len(STATE["pending"]):
            STATE["pending"].pop(int(idx))
        _save()
        return api_state()


@app.delete("/api/result/<int:index>")
def api_delete(index):
    with LOCK:
        opt = _opt()
        try:
            opt.remove_result(index)
        except IndexError as err:
            raise ApiError(str(err), 404)
        _save()
        return api_state()


@app.get("/api/model")
def api_model():
    """GP visualization data: 1D slices, factor importance, task correlation."""
    with LOCK:
        opt = _opt()
        if opt.num_results < 2:
            return jsonify({"available": False,
                            "reason": f"Add at least 2 results to see the model "
                                      f"(have {opt.num_results})."})
        try:
            gp = opt.fitted_gp()
        except Exception as err:  # noqa: BLE001 -- surface fit problems in the UI
            return jsonify({"available": False, "reason": f"GP fit failed: {err}"})

        ref = opt.best()["x"]
        grid_n = 60
        slices = []
        for j in range(opt.num_factors):
            lo, hi = opt.bounds[j].tolist()
            xs = torch.linspace(lo, hi, grid_n)
            X = ref.unsqueeze(0).repeat(grid_n, 1)
            X[:, j] = xs
            mean, lower, upper = gp.predict(X)
            slices.append({
                "grid": xs.tolist(),
                "mean": mean.T.tolist(),     # [task][point]
                "lower": lower.T.tolist(),
                "upper": upper.T.tolist(),
            })
        ls = torch.atleast_1d(gp.lengthscales.flatten())
        importance = (1.0 / ls)
        importance = (importance / importance.max()).tolist()
        C = gp.task_covariance
        d = C.diagonal().clamp_min(1e-12).sqrt()
        correlation = (C / torch.outer(d, d)).tolist()
        return jsonify({
            "available": True,
            "reference": ref.tolist(),
            "slices": slices,
            "importance": importance,
            "task_correlation": correlation,
            "predicted_best": _point(opt, opt.predicted_best()),
        })


@app.get("/api/export.csv")
def api_export():
    with LOCK:
        opt = _opt()
        buf = io.StringIO()
        writer = csv.writer(buf)
        header = ["experiment"] + opt.factor_names + opt.response_names
        priced = opt.costs is not None
        if priced:
            header.append(f"cost [{opt.costs.currency}]")
            costs = (opt.costs.experiment_cost(opt.train_x).tolist()
                     if opt.num_results else [])
        writer.writerow(header)
        for i in range(opt.num_results):
            row = ([i + 1]
                   + [f"{v:.6g}" for v in opt.train_x[i].tolist()]
                   + [f"{v:.6g}" for v in opt.train_y[i].tolist()])
            if priced:
                row.append(f"{costs[i]:.2f}")
            writer.writerow(row)
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition":
                                 "attachment; filename=doe_results.csv"})


def main():
    parser = argparse.ArgumentParser(
        description="Web UI for interactive DoE sessions.")
    parser.add_argument("session", nargs="?", default="session.json",
                        help="session file to resume or create (default: session.json)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    _load(args.session)
    resumed = (f"resumed, {STATE['opt'].num_results} results"
               if STATE["opt"] is not None else "new -- setup wizard in browser")
    print(f"DoE web UI:  http://localhost:{args.port}   (session {args.session}: {resumed})")
    print(f"Over ssh:    ssh -L {args.port}:localhost:{args.port} <user>@<this machine>")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
