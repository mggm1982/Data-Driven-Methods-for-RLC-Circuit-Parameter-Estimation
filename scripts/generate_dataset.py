"""Generate noiseless series-RLC metadata and physical-unit step responses."""
import argparse
import csv
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.integrate import solve_ivp

PARAMETER_SEED = 20261010
SPLIT_SEED = 20261011
COUNT = 10000
RANGES = {
    "resistance_ohm": (10.0, 100.0),
    "inductance_h": (0.010, 0.050),
    "capacitance_f": (1e-6, 10e-6),
}
COLUMNS = ["sample_id", *RANGES, "damping_ratio", "split"]
TIME = np.arange(5001, dtype=np.float64) * 1e-5


def sample_metadata():
    rng = np.random.default_rng(PARAMETER_SEED)
    parameters = [np.exp(rng.uniform(np.log(lo), np.log(hi), COUNT))
                  for lo, hi in RANGES.values()]
    r, l, c = parameters
    splits = np.array(["train"] * 7000 + ["validation"] * 1500 + ["test"] * 1500)
    np.random.default_rng(SPLIT_SEED).shuffle(splits)
    return [dict(zip(COLUMNS, (i, r[i], l[i], c[i],
                               r[i] / 2 * np.sqrt(c[i] / l[i]), splits[i])))
            for i in range(COUNT)]


def validate_metadata(rows):
    if len(rows) != COUNT:
        raise ValueError("Expected exactly 10000 circuits")
    ids = np.array([row["sample_id"] for row in rows])
    if not np.array_equal(np.sort(ids), np.arange(COUNT)):
        raise ValueError("IDs must be unique integers 0 through 9999")
    parameters = np.array([[row[key] for key in RANGES] for row in rows])
    if not np.isfinite(parameters).all() or len(np.unique(parameters, axis=0)) != COUNT:
        raise ValueError("Parameters must be finite and unique")
    for col, (lo, hi) in enumerate(RANGES.values()):
        if not ((parameters[:, col] >= lo) & (parameters[:, col] <= hi)).all():
            raise ValueError("Parameter outside agreed range")
    r, l, c = parameters.T
    damping = np.array([row["damping_ratio"] for row in rows])
    if not np.isfinite(damping).all() or not np.allclose(
            damping, r / 2 * np.sqrt(c / l), rtol=1e-14, atol=0):
        raise ValueError("Incorrect damping metadata")
    splits, counts = np.unique([row["split"] for row in rows], return_counts=True)
    split_counts = dict(zip(splits.tolist(), counts.tolist()))
    if split_counts != {"train": 7000, "validation": 1500, "test": 1500}:
        raise ValueError("Incorrect splits")
    return {"rows": len(rows), "unique_ids": len(np.unique(ids)),
            "unique_parameter_combinations": len(np.unique(parameters, axis=0)),
            "split_counts": split_counts,
            "damping_ratio_min": float(damping.min()),
            "damping_ratio_max": float(damping.max()),
            "underdamped": int(np.sum(damping < 1)),
            "critical": int(np.sum(damping == 1)),
            "overdamped": int(np.sum(damping > 1)),
            "near_critical_abs_zeta_minus_1_lt_0_01": int(np.sum(abs(damping - 1) < .01))}


def read_metadata(path):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise ValueError("Unexpected CSV columns")
        rows = [{key: (value if key == "split" else int(value) if key == "sample_id"
                       else float(value)) for key, value in row.items()} for row in reader]
    validate_metadata(rows)
    return rows


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def generate_metadata(directory):
    directory.mkdir(parents=True, exist_ok=True)
    rows = sample_metadata()
    validation = validate_metadata(rows)
    with (directory / "circuits.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    # Validate the persisted CSV, not just the in-memory sample.
    assert validate_metadata(read_metadata(directory / "circuits.csv")) == validation
    config = {
        "topology": "series RLC", "stimulus": {"type": "step", "before_t0_v": 0, "from_t0_v": 1},
        "initial_conditions": {"current_a": 0, "capacitor_voltage_v": 0},
        "equations": ["di/dt = (1 - R*i - v_C)/L", "dv_C/dt = i/C"],
        "units": {"resistance": "ohm", "inductance": "H", "capacitance": "F",
                  "time": "s", "current": "A", "capacitor_voltage": "V"},
        "parameter_ranges": {key: {"min": lo, "max": hi, "distribution": "log-uniform"}
                             for key, (lo, hi) in RANGES.items()},
        "sampling": "independent parameters; no damping stratification",
        "seeds": {"parameters": PARAMETER_SEED, "splits": SPLIT_SEED},
        "rng": "NumPy default_rng (PCG64)",
        "time_grid": {"start_s": 0, "stop_s": .050, "spacing_s": 1e-5, "points": 5001},
        "noise": False, "waveform_storage_dtype": "float32",
        "metadata_note": "R,L,C are labels; damping_ratio and this config are not model inputs",
        "metadata_validation": validation,
        "metadata_versions": {"python": platform.python_version(), "numpy": np.__version__},
    }
    write_json(directory / "dataset_config.json", config)
    return validation


def response(r, l, c, time):
    """Float64 closed-form solution, stable at/near critical damping.

    S solves the homogeneous impulse response; i=S/L and v_C=1-E-alpha*S.
    Overdamped roots use a cancellation-free slow root and expm1 difference.
    """
    alpha = r / (2 * l)
    omega_squared = 1 / (l * c)
    delta = omega_squared - alpha * alpha
    if delta > 0:
        omega = np.sqrt(delta)
        decay = np.exp(-alpha * time)
        s = decay * time * np.sinc(omega * time / np.pi)
        e = decay * np.cos(omega * time)
    elif delta < 0:
        beta = np.sqrt(-delta)
        slow = -omega_squared / (alpha + beta)
        fast = -alpha - beta
        s = np.exp(slow * time) * (-np.expm1(-2 * beta * time)) / (2 * beta)
        e = (np.exp(slow * time) + np.exp(fast * time)) / 2
    else:
        decay = np.exp(-alpha * time)
        s = time * decay
        e = decay
    return s / l, 1 - e - alpha * s


def reference(r, l, c, time):
    solution = solve_ivp(lambda t, y: [(1 - r * y[0] - y[1]) / l, y[0] / c],
                         (0, float(time[-1])), [0, 0], t_eval=time,
                         method="DOP853", rtol=1e-11, atol=1e-14)
    if not solution.success:
        raise ValueError(solution.message)
    return solution.y


def pilot_parameters(rows):
    # Every range corner, exact and near-critical cases, plus seeded CSV samples.
    corners = [(r, l, c) for r in RANGES["resistance_ohm"]
               for l in RANGES["inductance_h"] for c in RANGES["capacitance_f"]]
    l, c = .01, 1e-5
    critical = 2 * np.sqrt(l / c)
    cases = corners + [(critical * factor, l, c) for factor in (1, 1 - 1e-10, 1 + 1e-10)]
    indices = np.random.default_rng(20261012).choice(len(rows), 12, replace=False)
    cases += [tuple(rows[i][key] for key in RANGES) for i in indices]
    return cases


def validate_pilot(rows):
    maxima = np.zeros(6)
    fine_time = np.arange(50001, dtype=np.float64) * 1e-6
    cases = pilot_parameters(rows)
    for r, l, c in cases:
        i, v = response(r, l, c, TIME)
        ref_i, ref_v = reference(r, l, c, TIME)
        fine_i, fine_v = response(r, l, c, fine_time)
        errors = [np.max(abs(i.astype(np.float32) - ref_i)),
                  np.max(abs(v.astype(np.float32) - ref_v)),
                  np.max(abs(np.interp(fine_time, TIME, i) - fine_i)) / np.max(abs(fine_i)),
                  np.max(abs(np.interp(fine_time, TIME, v) - fine_v)),
                  abs(i[-1]), abs(v[-1] - 1)]
        maxima = np.maximum(maxima, errors)
    if maxima[0] > 1e-8 or maxima[1] > 1e-6:
        raise ValueError("Pilot disagrees with independent ODE reference")
    if maxima[2] > .005 or maxima[3] > .005:
        raise ValueError("Time grid exceeds 0.5% pilot interpolation-error limit")
    return dict(zip(["max_reference_current_error_a", "max_reference_voltage_error_v",
                     "max_linear_interpolation_current_error_relative_to_peak",
                     "max_linear_interpolation_voltage_error_v",
                     "max_final_abs_current_a", "max_final_abs_voltage_minus_1_v"],
                    maxima.tolist()), cases=len(cases), fine_spacing_s=1e-6,
                reference_method="solve_ivp DOP853", reference_rtol=1e-11, reference_atol=1e-14,
                note="Reference errors include float32 storage; interpolation compares to a 10x finer analytical grid. Settling residuals are measured, not required to vanish.")


def generate_waveforms(directory):
    rows = read_metadata(directory / "circuits.csv")
    # Fail before allocating/generating the full dataset if the pilot fails.
    pilot = validate_pilot(rows)
    print("Pilot:", json.dumps(pilot, indent=2), flush=True)
    current = np.empty((len(rows), len(TIME)), dtype=np.float32)
    voltage = np.empty_like(current)
    for index, row in enumerate(rows):
        current[index], voltage[index] = response(*(row[key] for key in RANGES), TIME)
    ids = np.array([row["sample_id"] for row in rows], dtype=np.int64)
    path = directory / "waveforms.npz"
    np.savez(path, sample_id=ids, time_s=TIME, current_a=current, capacitor_voltage_v=voltage)
    del current, voltage
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {"sample_id", "time_s", "current_a", "capacitor_voltage_v"}:
            raise ValueError("Unexpected NPZ keys")
        if not np.array_equal(saved["sample_id"], ids) or not np.array_equal(saved["time_s"], TIME):
            raise ValueError("Saved ID/time alignment mismatch")
        current, voltage = saved["current_a"], saved["capacitor_voltage_v"]
        for signals in (current, voltage):
            if signals.shape != (COUNT, 5001) or signals.dtype != np.float32:
                raise ValueError("Incorrect waveform shape/dtype")
            if not np.isfinite(signals).all() or not (signals[:, 0] == 0).all():
                raise ValueError("Nonfinite waveform or incorrect initial state")
        # Independent reference checks on representative actual persisted rows.
        indices = [int(np.argmin([row["damping_ratio"] for row in rows])),
                   int(np.argmax([row["damping_ratio"] for row in rows])),
                   int(np.argmin([abs(row["damping_ratio"] - 1) for row in rows]))]
        indices += np.random.default_rng(20261013).choice(COUNT, 12, replace=False).tolist()
        ref_errors = np.zeros(2)
        for index in indices:
            ref = reference(*(rows[index][key] for key in RANGES), TIME)
            ref_errors = np.maximum(ref_errors, [np.max(abs(current[index] - ref[0])),
                                                 np.max(abs(voltage[index] - ref[1]))])
        if ref_errors[0] > 1e-8 or ref_errors[1] > 1e-6:
            raise ValueError("Saved waveforms disagree with reference")
        # At 200 ms even the slowest allowed envelope has decayed by exp(-20).
        long_residual = np.zeros(2)
        for row in rows:
            i, v = response(*(row[key] for key in RANGES), np.array([.2]))
            long_residual = np.maximum(long_residual, [abs(i[0]), abs(v[0] - 1)])
        if long_residual[0] > 1e-8 or long_residual[1] > 1e-7:
            raise ValueError("Incorrect long-time steady state")
        validation = {
            "rows_checked": COUNT, "shape_per_signal": list(current.shape),
            "finite_and_zero_initial_states": True, "csv_id_order_matches": True,
            "reference_rows_checked": len(indices), "reference_row_indices": indices,
            "max_reference_current_error_a": float(ref_errors[0]),
            "max_reference_voltage_error_v": float(ref_errors[1]),
            "max_final_abs_current_a": float(np.max(abs(current[:, -1]))),
            "max_final_abs_voltage_minus_1_v": float(np.max(abs(voltage[:, -1] - 1))),
            "steady_state_check_time_s": .2,
            "max_steady_state_abs_current_a": float(long_residual[0]),
            "max_steady_state_abs_voltage_minus_1_v": float(long_residual[1]),
            "npz_bytes": path.stat().st_size,
        }
    config_path = directory / "dataset_config.json"
    config = json.loads(config_path.read_text())
    config.update({"simulation": {"method": "closed-form analytical step response; sinc underdamped, expm1/root form overdamped, critical limit",
                                   "computation_dtype": "float64", "storage_dtype": "float32",
                                   "compression": False,
                                   "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
                                   "pilot_sample_seed": 20261012, "saved_reference_sample_seed": 20261013},
                   "pilot_validation": pilot, "waveform_validation": validation})
    write_json(config_path, config)
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["metadata", "pilot", "waveforms"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.command == "metadata":
        result = generate_metadata(args.data_dir)
    elif args.command == "pilot":
        result = validate_pilot(read_metadata(args.data_dir / "circuits.csv"))
    else:
        result = generate_waveforms(args.data_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
