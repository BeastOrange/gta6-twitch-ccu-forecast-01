"""Forecast GTA VI's launch-day Twitch CCU hour by hour (make.fun gta6-twitch-ccu-forecast-01).

Model: an empirical forecaster learned at run time from the case file's own
`comparables` (past big launches' hourly Twitch curves, SullyGnome).

For each past launch we know its release-day curve: hour 1's print and hours
2..24.  What varies most across launches is not the shape of the day but how
much of the launch was already unlocked in hour 1: a game whose hour 1 was
still pre-unlock in most time zones grows 100x by the evening peak, while a
game whose hour 1 already caught a midnight unlock barely grows at all
(log hour-1 viewers vs log peak/hour-1 correlates at -0.94 across the 37
usable comparables).  So the forecast conditions on the size of hour 1:

* kernel component: for each hour, the quantiles of log(viewers_h / viewers_1)
  across comparables, weighted by a Gaussian kernel in log(hour-1 viewers)
  (bandwidth 1.0), so a huge hour 1 borrows the shape of the biggest past
  launches and a tiny hour 1 borrows the shape of the slow starters;
* regression component: per-hour OLS of log(viewers_h / viewers_1) on centred
  log(hour-1 viewers), with the predictor capped at its 90th percentile so a
  record-breaking hour 1 is never extrapolated linearly past the data, and
  empirical residual quantiles for the spread.

The entry is the geometric mean (average in log space) of the two components'
quantiles: the kernel adapts to the launch's scale, the regression hedges it
toward the global trend.  Quantiles are enforced non-decreasing and rounded
half-up to 2 decimals, as the starter does.

Leave-one-out validation on the 37 usable comparables (mean pinball loss on
the log scale, lower better; house baseline = flat at hour 1, sigma 0.15):

    house baseline .......... 0.930  (0.261 on launches with hour 1 > 100k)
    this model .............. 0.387  (0.221 on launches with hour 1 > 100k)

The case's own `history` is not used for the shape: pre-release levels add
nothing once hour 1 is known (R^2 ~ 0.04 for the peak), so it serves only as
a fallback anchor if a case ever arrived without an hour-1 print.  `streams`
and `categories_hourly` (null in the practice cases) are likewise unused.

Runs offline, standard library only, deterministic (sorted inputs, no
randomness, no wall-clock or hash-order dependence), far inside the 8 s cap.

Usage (outside the grader):  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves the program's directory OFF sys.path: put it back
# first, so a module shipped beside this file would import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (house baseline's spread).
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# House baseline's spread, in natural-log units per square-root hour.
SIGMA = 0.15
# Gaussian kernel bandwidth for the scale-conditioned component, in log units.
BANDWIDTH = 1.0
# The regression predictor is capped at this percentile of training log hour-1.
CAP_PERCENTILE = 0.9
# Minimum usable comparables before falling back to the house baseline.
MIN_GAMES = 5
# A comparable needs at least this many release-day hours to count.
MIN_HOURS = 12


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_day_hours(hourly, day_start):
    """{hour 1..24: viewers} for the UTC day starting at day_start."""
    hours = {}
    for point in hourly or []:
        try:
            when = parse_ts(point["from"])
            viewers = point["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        h = int((when - day_start).total_seconds() // 3600) + 1
        if 1 <= h <= 24 and isinstance(viewers, (int, float)):
            hours[h] = viewers
    return hours


def build_training(comparables):
    """Usable past launches: (hour-1 viewers, {hour: viewers}) per launch."""
    training = []
    for comp in comparables or []:
        try:
            day_start = parse_ts(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        hours = release_day_hours(comp.get("hourly"), day_start)
        h1 = hours.get(1)
        if not h1 or h1 <= 0:
            continue
        if sum(1 for h in range(2, 25) if h in hours and hours[h] > 0) < MIN_HOURS:
            continue
        training.append((h1, hours))
    return training


def quantile_interp(sorted_values, tau):
    """Linear-interpolation quantile of a sorted list (numpy's default rule)."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    pos = tau * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def weighted_quantile(pairs, tau):
    """Step weighted quantile: pairs are (value, weight), sorted by value.

    Returns the first value whose cumulative weight reaches tau of the total.
    """
    total = sum(w for _, w in pairs)
    target = tau * total
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= target:
            return value
    return pairs[-1][0]


def kernel_quantiles(training, log_p1, hour):
    """Scale-conditioned quantiles of log(viewers_h / viewers_1) at one hour."""
    pairs = []
    for h1, hours in training:
        v = hours.get(hour)
        if v is None or v <= 0:
            continue
        weight = math.exp(-0.5 * ((math.log(h1) - log_p1) / BANDWIDTH) ** 2)
        pairs.append((math.log(v / h1), weight))
    if len(pairs) < 3:
        return None
    pairs.sort(key=lambda p: p[0])
    return [weighted_quantile(pairs, tau) for tau in LEVELS]


def regression_fit(training):
    """Per-hour OLS of log-ratio on centred log hour-1, plus residual quantiles.

    Returns {hour: (intercept, slope, centre, cap, [residual quantiles])}.
    The predictor is capped at its CAP_PERCENTILE so a record hour 1 is
    never extrapolated linearly past the observed range.
    """
    log_h1s = sorted(math.log(h1) for h1, _ in training)
    centre = sum(log_h1s) / len(log_h1s)
    cap = log_h1s[int(CAP_PERCENTILE * (len(log_h1s) - 1))]
    fit = {}
    for hour in range(2, 25):
        points = [
            (math.log(h1), math.log(hours[hour] / h1))
            for h1, hours in training
            if hours.get(hour) and hours[hour] > 0
        ]
        if len(points) < 3:
            continue
        xs = [x - centre for x, _ in points]
        ys = [y for _, y in points]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        denom = sum((x - mean_x) ** 2 for x in xs)
        slope = (
            sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
            if denom > 0
            else 0.0
        )
        intercept = mean_y - slope * mean_x
        residuals = sorted(y - (intercept + slope * x) for x, y in zip(xs, ys))
        fit[hour] = (
            intercept,
            slope,
            centre,
            cap,
            [quantile_interp(residuals, tau) for tau in LEVELS],
        )
    return fit


def regression_quantiles(fit, log_p1, hour):
    entry = fit.get(hour)
    if entry is None:
        return None
    intercept, slope, centre, cap, residual_qs = entry
    x = min(log_p1 - centre, cap - centre)
    median = intercept + slope * x
    return [median + r for r in residual_qs]


def house_quantiles(p1, hour):
    """The house baseline: flat at hour 1, spread widening like a random walk."""
    return [p1 * math.exp(z * SIGMA * math.sqrt(hour - 1)) for z in Z]


def forecast_case(case, name):
    levels = case.get("levels") or LEVELS
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is not None and first.get("viewers"):
        p1 = max(1, first["viewers"])
    else:
        # No hour-1 print: fall back to the last pre-checkpoint hourly point.
        hourly = case.get("history_hourly") or []
        p1 = max(1, hourly[-1]["viewers"]) if hourly else 1
    log_p1 = math.log(p1)

    training = build_training(case.get("comparables"))
    fit = regression_fit(training) if len(training) >= MIN_GAMES else {}

    hours_out = []
    for row in case["forecast"]:
        hour = row["hour"]
        kernel = (
            kernel_quantiles(training, log_p1, hour)
            if len(training) >= MIN_GAMES
            else None
        )
        regress = regression_quantiles(fit, log_p1, hour) if fit else None
        if kernel is not None and regress is not None:
            # Geometric mean of the two components, level by level.
            log_qs = [
                (k + r) / 2.0 for k, r in zip(kernel, regress)
            ]
            qs = [p1 * math.exp(lq) for lq in log_qs]
        elif kernel is not None:
            qs = [p1 * math.exp(lq) for lq in kernel]
        elif regress is not None:
            qs = [p1 * math.exp(lq) for lq in regress]
        else:
            qs = house_quantiles(p1, hour)
        # Safety net: non-decreasing, positive, within the contest's bounds.
        fixed = []
        prev = 0.0
        for q in qs:
            q = min(max(q, prev), 1e9)
            q = max(q, 1e-6)
            fixed.append(round2(q))
            prev = fixed[-1]
        hours_out.append({"hour": hour, "q": fixed})
    return {"hours": hours_out}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
