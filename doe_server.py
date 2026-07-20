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
import re
import threading
from pathlib import Path

import torch
from flask import Flask, Response, jsonify, request, send_from_directory

from doe import IngredientCosts, InteractiveOptimizer

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(ROOT / "webui"), static_url_path="/static")

LOCK = threading.RLock()
STATE = {"opt": None, "path": "session.json", "pending": [], "desc": ""}

# Remembers which session the UI selected, so it survives server restarts
# (systemd always passes the same session argument).
ACTIVE_MARKER = ".active_session"

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$")


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
    """Persist the session; description + pending proposals ride along."""
    opt = STATE["opt"]
    opt.meta["description"] = STATE["desc"]
    opt.meta["pending"] = STATE["pending"]
    opt.save(STATE["path"])


def _load(path):
    """Make ``path`` the active session.

    The file may not exist yet, or may be an unconfigured stub holding just a
    description -- both show the setup wizard. STATE is only touched once the
    file parsed, so a corrupt file leaves the current session active.
    """
    p = Path(path)
    opt, pending, desc = None, [], ""
    if p.is_file():
        with open(p) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("not a DoE session file")
        if "bounds" in data:
            opt = InteractiveOptimizer.load(p)
            pending = opt.meta.get("pending", [])
            desc = str(opt.meta.get("description", ""))
        else:
            desc = str(data.get("description", ""))
    STATE.update(path=str(path), opt=opt, pending=pending, desc=desc)
    STATE.pop("predicted_best", None)  # cached GP belongs to the old session


def _sessions_dir():
    return Path(STATE["path"]).resolve().parent


def _session_path(name, must_exist=False, must_not_exist=False):
    """Validate a user-supplied session name -> path inside the sessions dir.

    Names come from the network, so this is the only gate between the API and
    the filesystem: one flat directory, ``<safe name>.json`` only.
    """
    base = str(name or "").strip()
    if base.endswith(".json"):
        base = base[:-5].strip()
    if not _NAME_RE.match(base):
        raise ApiError("session name must start with a letter or digit and "
                       "may only contain letters, digits, spaces and . _ - "
                       "(at most 64 characters)")
    directory = _sessions_dir()
    path = (directory / f"{base}.json").resolve()
    if path.parent != directory:  # belt and braces; the regex has no separators
        raise ApiError("invalid session name")
    if must_exist and not path.is_file():
        raise ApiError(f"no session named '{base}'", 404)
    if must_not_exist and path.exists():
        raise ApiError(f"a session named '{base}' already exists", 409)
    return path


def _is_active(path):
    return path == Path(STATE["path"]).resolve()


def _write_marker():
    try:
        (_sessions_dir() / ACTIVE_MARKER).write_text(Path(STATE["path"]).name)
    except OSError:
        pass  # read-only dir: switching still works, it just won't persist


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


def _predicted_best(opt):
    """predicted_best() searches a fresh random candidate pool each call;
    cache one result per fitted GP so every tab shows the same optimum."""
    gp = opt.fitted_gp()
    cached = STATE.get("predicted_best")
    if cached is None or cached[0] is not gp:
        STATE["predicted_best"] = (gp, opt.predicted_best())
    return STATE["predicted_best"][1]


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/state")
def api_state():
    with LOCK:
        opt = STATE["opt"]
        if opt is None:
            return jsonify({"configured": False, "session": STATE["path"],
                            "description": STATE["desc"]})
        resp = {
            "configured": True,
            "session": STATE["path"],
            "description": STATE["desc"],
            "factor_names": opt.factor_names,
            "response_names": opt.response_names,
            "bounds": opt.bounds.tolist(),
            "target_task": opt.target_task,
            "maximize": opt.maximize,
            "cost_aware": opt.cost_aware,
            "num_init": opt.num_init,
            "kernel": opt.kernel,
            "poly_degree": opt.poly_degree,
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
            raise ApiError("session already configured -- create a new "
                           "session in the session manager for a fresh DoE", 409)
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
        STATE.pop("predicted_best", None)
        _save()
        return api_state()


@app.post("/api/fit")
def api_fit():
    """Choose how the posterior is fit: flexible GP (rbf) or n-degree polynomial."""
    with LOCK:
        opt = _opt()
        data = request.get_json(force=True)
        try:
            opt.set_kernel(str(data.get("kernel", "rbf")),
                           data.get("degree"))
        except (ValueError, TypeError) as err:
            raise ApiError(str(err))
        STATE.pop("predicted_best", None)  # cached optimum used the old fit
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
        ls = gp.lengthscales  # None for the polynomial kernel (no ARD)
        if ls is None:
            importance = None
        else:
            ls = torch.atleast_1d(ls.flatten())
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
            "predicted_best": _point(opt, _predicted_best(opt)),
        })


@app.get("/api/landscape")
def api_landscape():
    """GP response surface over every factor pair (corner-plot data).

    For each pair (i, j) the remaining factors are held at the best measured
    experiment (same reference as /api/model), and the GP posterior is
    evaluated on a grid_n x grid_n grid: ``mean[task][row][col]`` where rows
    follow factor j and columns factor i.
    """
    with LOCK:
        opt = _opt()
        if opt.num_factors < 2:
            return jsonify({"available": False,
                            "reason": "The landscape shows ingredient pairs -- "
                                      "this session has only one ingredient."})
        if opt.num_results < 2:
            return jsonify({"available": False,
                            "reason": f"Add at least 2 results to see the "
                                      f"landscape (have {opt.num_results})."})
        try:
            gp = opt.fitted_gp()
        except Exception as err:  # noqa: BLE001 -- surface fit problems in the UI
            return jsonify({"available": False, "reason": f"GP fit failed: {err}"})

        ref = opt.best()["x"]
        F = opt.num_factors
        grid_n = 40 if F == 2 else 28 if F <= 4 else 18
        pairs = []
        for i in range(F):
            for j in range(i + 1, F):
                gi = torch.linspace(*opt.bounds[i].tolist(), grid_n)
                gj = torch.linspace(*opt.bounds[j].tolist(), grid_n)
                X = ref.unsqueeze(0).repeat(grid_n * grid_n, 1)
                X[:, i] = gi.repeat(grid_n)                # columns sweep i
                X[:, j] = gj.repeat_interleave(grid_n)     # rows sweep j
                # predictive distribution (incl. observation noise) -- the
                # same uncertainty convention as the /api/model bands
                mean, lower, upper = gp.predict(X)
                mean = mean.reshape(grid_n, grid_n, opt.num_tasks)
                std = ((upper - lower) / 4.0).reshape(   # 95% band = +-2 sigma
                    grid_n, grid_n, opt.num_tasks)
                pairs.append({
                    "i": i, "j": j,
                    "gi": gi.tolist(), "gj": gj.tolist(),
                    "mean": mean.permute(2, 0, 1).tolist(),  # [task][row][col]
                    "std": std.permute(2, 0, 1).tolist(),
                })
        return jsonify({
            "available": True,
            # session + result count + fit choice let the client detect a
            # payload that belongs to another session or data version (the
            # server is shared -- a second window can switch it under us)
            "session": STATE["path"],
            "n": opt.num_results,
            "kernel": opt.kernel,
            "poly_degree": opt.poly_degree,
            "reference": ref.tolist(),
            "pairs": pairs,
            "predicted_best": _point(opt, _predicted_best(opt)),
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
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(STATE["path"]).stem)
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition":
                                 f"attachment; filename={stem}_results.csv"})


# ------------------------------------------------------- session management


@app.get("/api/sessions")
def api_sessions():
    """All session files in the sessions directory, newest first."""
    with LOCK:
        entries = []
        for p in _sessions_dir().glob("*.json"):
            try:
                with open(p) as f:
                    data = json.load(f)
                modified = p.stat().st_mtime
            except (OSError, ValueError):
                continue  # unreadable or not JSON -- not one of ours
            if not isinstance(data, dict):
                continue
            configured = "bounds" in data
            if not (configured or data.get("doe_session")):
                continue  # some unrelated JSON file living in the directory
            active = _is_active(p.resolve())
            entries.append({
                "name": p.name,
                "description": (STATE["desc"] if active
                                else str(data.get("description", ""))),
                "configured": configured,
                "num_results": len(data.get("train_x", [])),
                "factor_names": data.get("factor_names", []),
                "response_names": data.get("response_names", []),
                "modified": modified,
                "active": active,
            })
        if not any(e["active"] for e in entries):
            # active session not saved to disk yet (fresh, unconfigured)
            entries.append({
                "name": Path(STATE["path"]).name, "description": STATE["desc"],
                "configured": STATE["opt"] is not None,
                "num_results": 0, "factor_names": [], "response_names": [],
                "modified": None, "active": True,
            })
        entries.sort(key=lambda e: e["modified"] or math.inf, reverse=True)
        return jsonify({"dir": str(_sessions_dir()), "sessions": entries})


@app.post("/api/sessions/create")
def api_sessions_create():
    """Create a new (empty) session file and switch to it."""
    with LOCK:
        data = request.get_json(force=True)
        path = _session_path(data.get("name"), must_not_exist=True)
        desc = str(data.get("description", "")).strip()
        with open(path, "w") as f:
            json.dump({"doe_session": True, "description": desc}, f, indent=2)
        _load(path)
        _write_marker()
        return api_state()


@app.post("/api/sessions/load")
def api_sessions_load():
    with LOCK:
        path = _session_path(request.get_json(force=True).get("name"),
                             must_exist=True)
        try:
            _load(path)
        except (ValueError, KeyError, TypeError) as err:
            raise ApiError(f"cannot load {path.name}: {err}")
        _write_marker()
        return api_state()


@app.post("/api/sessions/rename")
def api_sessions_rename():
    with LOCK:
        data = request.get_json(force=True)
        src = _session_path(data.get("name"), must_exist=True)
        dst = _session_path(data.get("new_name"), must_not_exist=True)
        active = _is_active(src)
        src.rename(dst)
        if active:
            STATE["path"] = str(dst)
            _write_marker()
        return jsonify({"ok": True, "name": dst.name})


@app.post("/api/sessions/delete")
def api_sessions_delete():
    with LOCK:
        path = _session_path(request.get_json(force=True).get("name"),
                             must_exist=True)
        if _is_active(path):
            raise ApiError("cannot delete the active session -- "
                           "load a different one first", 409)
        path.unlink()
        return jsonify({"ok": True})


@app.post("/api/sessions/describe")
def api_sessions_describe():
    with LOCK:
        data = request.get_json(force=True)
        path = _session_path(data.get("name"))
        desc = str(data.get("description", "")).strip()
        if _is_active(path):
            STATE["desc"] = desc
            if STATE["opt"] is not None:
                _save()
                return jsonify({"ok": True})
            # unconfigured: fall through to patch the stub file, if any
        elif not path.is_file():
            raise ApiError(f"no session named '{path.stem}'", 404)
        if path.is_file():
            try:
                with open(path) as f:
                    file_data = json.load(f)
            except ValueError:
                raise ApiError(f"{path.name} is not a valid session file")
            if not isinstance(file_data, dict):
                raise ApiError(f"{path.name} is not a DoE session file")
            file_data["description"] = desc
            with open(path, "w") as f:
                json.dump(file_data, f, indent=2)
        return jsonify({"ok": True})


def main():
    parser = argparse.ArgumentParser(
        description="Web UI for interactive DoE sessions.")
    parser.add_argument("session", nargs="?", default="session.json",
                        help="session file to resume or create (default: session.json)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # The session picked in the web UI wins over the command-line default,
    # so a UI switch survives restarts. Delete the marker file to override.
    session = Path(args.session)
    marker = session.resolve().parent / ACTIVE_MARKER
    if marker.is_file():
        name = marker.read_text().strip()
        last = session.resolve().parent / name
        if name and last.is_file() and last != session.resolve():
            print(f"Resuming last active session {name} "
                  f"(picked in the web UI; rm {ACTIVE_MARKER} to override)")
            session = last
    try:
        _load(session)
    except (ValueError, KeyError, TypeError) as err:
        if session == Path(args.session):
            raise SystemExit(f"cannot load {session}: {err}")
        print(f"cannot load {session.name} ({err}) -- "
              f"falling back to {args.session}")
        _load(args.session)
    _write_marker()
    resumed = (f"resumed, {STATE['opt'].num_results} results"
               if STATE["opt"] is not None else "new -- setup wizard in browser")
    print(f"DoE web UI:  http://localhost:{args.port}   (session {Path(STATE['path']).name}: {resumed})")
    print(f"Over ssh:    ssh -L {args.port}:localhost:{args.port} <user>@<this machine>")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
