"""Interactive DoE session: propose experiments, enter lab results, adapt.

Usage
-----
    .venv/bin/python interactive_doe.py session.json

If the session file exists it is resumed; otherwise a short setup wizard asks
for the ingredients (with bounds and prices), the responses, and the target.
Every entered result is saved immediately, so the session can be closed and
resumed between lab days.
"""

import sys

from doe import InteractiveOptimizer, IngredientCosts


def _ask(prompt, cast=str, default=None):
    while True:
        raw = input(f"{prompt}{f' [{default}]' if default is not None else ''}: ").strip()
        if not raw and default is not None:
            return default
        try:
            return cast(raw)
        except ValueError:
            print("  invalid value, try again")


def _ask_floats(prompt, n):
    while True:
        raw = input(f"{prompt}: ").strip().replace(",", " ")
        parts = raw.split()
        if len(parts) == n:
            try:
                return [float(p) for p in parts]
            except ValueError:
                pass
        print(f"  expected {n} numbers separated by spaces")


def setup_wizard():
    print("=== New DoE session ===")
    num_factors = _ask("Number of ingredients (factors)", int)
    names, bounds, prices = [], [], []
    for i in range(num_factors):
        name = _ask(f"  Ingredient {i + 1} name", str, f"ingredient {i + 1}")
        lo, hi = _ask_floats(f"  {name}: min and max concentration", 2)
        price = _ask(f"  {name}: price per concentration unit", float, 0.0)
        names.append(name)
        bounds.append([lo, hi])
        prices.append(price)
    fixed = _ask("Fixed cost per experiment (overhead)", float, 0.0)
    currency = _ask("Currency", str, "USD")

    num_tasks = _ask("Number of measured responses (criteria)", int, 2)
    response_names = [
        _ask(f"  Response {t + 1} name", str, f"response {t + 1}")
        for t in range(num_tasks)
    ]
    target = _ask(f"Which response to optimize (1..{num_tasks})", int, 1) - 1
    maximize = _ask("Maximize it? (y/n)", str, "y").lower().startswith("y")
    cost_aware = _ask("Prefer cheaper experiments when proposing? (y/n)",
                      str, "n").lower().startswith("y")

    return InteractiveOptimizer(
        bounds=bounds, num_tasks=num_tasks, target_task=target,
        maximize=maximize, cost_aware=cost_aware,
        costs=IngredientCosts(prices, fixed_cost=fixed, names=names,
                              currency=currency),
        factor_names=names, response_names=response_names,
    )


def show_proposals(opt, proposals):
    for i, x in enumerate(proposals):
        settings = ", ".join(f"{n}={v.item():.4g}"
                             for n, v in zip(opt.factor_names, x))
        line = f"  [{i + 1}] {settings}"
        if opt.costs is not None:
            line += (f"   (~{opt.costs.experiment_cost(x).item():.2f} "
                     f"{opt.costs.currency})")
        print(line)


def show_best(opt):
    best = opt.best()
    if best is None:
        print("No results yet.")
        return
    print(f"Best measured so far (experiment #{best['index'] + 1}):")
    print("  " + ", ".join(f"{n}={v.item():.4g}"
                           for n, v in zip(opt.factor_names, best["x"])))
    print("  " + ", ".join(f"{n}={v.item():.4g}"
                           for n, v in zip(opt.response_names, best["y"])))
    if "cost" in best:
        cur = opt.costs.currency
        print(f"  experiment cost: {best['cost']:.2f} {cur}")
        if "cost_per_yield" in best:
            print(f"  price of the yield: {best['cost_per_yield']:.4g} {cur} "
                  f"per unit {opt.response_names[opt.target_task]}")

    pred = opt.predicted_best()
    if pred is not None and opt.num_results >= opt.num_init:
        print("Model's predicted optimum (not yet run):")
        print("  " + ", ".join(f"{n}={v.item():.4g}"
                               for n, v in zip(opt.factor_names, pred["x"])))
        print("  " + ", ".join(f"{n}~{v.item():.4g}"
                               for n, v in zip(opt.response_names, pred["y"])))
        if "cost_per_yield" in pred:
            print(f"  predicted price of the yield: "
                  f"{pred['cost_per_yield']:.4g} {opt.costs.currency} per unit "
                  f"{opt.response_names[opt.target_task]}")


def show_costs(opt):
    if opt.costs is None:
        print("No ingredient prices configured for this session.")
        return
    cur = opt.costs.currency
    print(f"Spent so far ({opt.num_results} experiments): "
          f"{opt.spend():.2f} {cur}")
    for n in (5, 10, 20):
        print(f"  projected total after {n} more experiments: "
              f"~{opt.projected_spend(n):.2f} {cur}")


def enter_result(opt, x):
    prompt = ", ".join(opt.response_names)
    y = _ask_floats(f"    measured {prompt}", opt.num_tasks)
    opt.add_result(x, y)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    try:
        opt = InteractiveOptimizer.load(path)
        print(f"Resumed session with {opt.num_results} results from {path}")
    except FileNotFoundError:
        opt = setup_wizard()
        opt.save(path)
        print(f"Session saved to {path}")

    pending = []
    print("\nCommands: [s]uggest, [r]esult, [b]est, [c]osts, [q]uit")
    while True:
        cmd = input("\n> ").strip().lower()
        if cmd.startswith("s"):
            n = _ask("How many experiments to propose (1-3)", int, 1)
            n = max(1, min(3, n))
            print("Proposed next experiments"
                  + (" (space-filling initial design)"
                     if opt.num_results < opt.num_init else " (GP-guided)") + ":")
            pending = list(opt.suggest(n))
            show_proposals(opt, pending)
            print("Run them in the lab, then enter results with [r].")
        elif cmd.startswith("r"):
            if pending:
                for i, x in enumerate(pending):
                    settings = ", ".join(f"{n}={v.item():.4g}"
                                         for n, v in zip(opt.factor_names, x))
                    print(f"  proposal [{i + 1}] {settings}")
                    raw = input("    enter results? (y to enter / n to skip): ")
                    if raw.strip().lower().startswith("y"):
                        enter_result(opt, x)
                pending = []
            else:
                print("  Manual entry -- concentrations of an experiment you ran:")
                x = _ask_floats("    " + ", ".join(opt.factor_names),
                                opt.num_factors)
                enter_result(opt, x)
            opt.save(path)
            print(f"  {opt.num_results} results total, session saved.")
        elif cmd.startswith("b"):
            show_best(opt)
        elif cmd.startswith("c"):
            show_costs(opt)
        elif cmd.startswith("q"):
            opt.save(path)
            print(f"Session saved to {path}")
            break
        else:
            print("Commands: [s]uggest, [r]esult, [b]est, [c]osts, [q]uit")


if __name__ == "__main__":
    main()
