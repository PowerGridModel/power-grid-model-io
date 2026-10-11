# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import csv
from pathlib import Path
from typing import TypedDict

import numpy as np
from power_grid_model import (
    AttributeType,
    CalculationMethod,
    CalculationType,
    ComponentType,
    DatasetType,
    LoadGenType,
    PowerGridModel,
    WindingType,
    initialize_array,
)
from power_grid_model.data_types import Dataset
from power_grid_model.errors import PowerGridError
from power_grid_model.utils import json_serialize_to_file
from power_grid_model.validation import assert_valid_input_data

root = Path(__file__).parent.parent.parent
DATA_DIR = root / "src" / "power_grid_model_io" / "networks" / "_data" / "ieee" / "ieee_four_bus"

curr_directory = Path(__file__).parent

WIRE_PARAM_FILE = curr_directory / "wire_data_4_bus.csv"

MAX_U_MAG_TOL = 5.0  # Max allowed voltage magnitude error (Volts)
MAX_U_ANGLE_TOL = 1  # Max allowed voltage phase angle error (Degrees)
MAX_I_MAG_TOL = 5.0  # Max allowed current magnitude error (Amperes)
MAX_I_ANGLE_TOL = 1.5  # Max allowed current phase angle error (Degrees)


class ScenarioConfig(TypedDict):
    u_primary: float
    u_secondary: float
    winding_from: WindingType
    winding_to: WindingType
    clock: int
    p_phases: list[float]
    pf_phases: list[float]
    expected_results: dict[str, np.ndarray]


winding_from_lookup = {
    "ynyn": WindingType.wye_n,
    "ynd": WindingType.wye_n,
    "yd": WindingType.wye,
    "dyn": WindingType.delta,
    "dd": WindingType.delta,
}

winding_to_lookup = {
    "ynyn": WindingType.wye_n,
    "ynd": WindingType.delta,
    "yd": WindingType.delta,
    "dyn": WindingType.wye_n,
    "dd": WindingType.delta,
}


license_content = (
    "# SPDX-FileCopyrightText: Contributors to the Power Grid Model project \n#\n# SPDX-License-Identifier: MPL-2.0\n"
)

MILES_TO_FEET = 5.280e3

p_bal = [1.8e6, 1.8e6, 1.8e6]
pf_bal = [0.9, 0.9, 0.9]

p_un_bal = [1.275e6, 1.8e6, 2.375e6]
pf_un_bal = [0.85, 0.9, 0.95]


def load_wire_params_dict(file_path: Path) -> dict[str, dict[str, float]]:
    wire_params = {}

    with file_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wire_type = row.pop("wire_type")
            wire_params[wire_type] = {k: float(v) for k, v in row.items()}

    return wire_params


def load_expected_results(csv_path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Loads CSV results into structured NumPy arrays for 'u' and 'i'.

    Output structure:
        results = {
            "ynyn": {
                "u": np.ndarray(dtype=[("id", int), ("v", "(3,)complex128")]),
                "i": np.ndarray(dtype=[("id", "U10"), ("i", "(3,)complex128")]),
            },
            ...
        }
    """
    results = {}

    # Define structured array dtypes with embedded (3,) 3-phase complex vectors
    u_dtype = np.dtype([("id", np.int32), ("v", "(3,)complex128")])
    i_dtype = np.dtype([("id", np.int32), ("i", "(3,)complex128")])

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            conn = row.pop("Connection").strip()

            v_cols = [k for k in row if k.startswith("V_")]
            i_cols = [k for k in row if k.startswith("I_")]

            node_ids = sorted(list({int(k.split("_")[1]) for k in v_cols}))
            branch_ids = sorted(list({k.split("_")[1] for k in i_cols}))

            # 1. Build Node Voltage Array (u)
            u_arr = np.empty(len(node_ids), dtype=u_dtype)
            for idx, node_id in enumerate(node_ids):
                phasors = []
                for phase in ("a", "b", "c"):
                    mag_str, ang_str = row[f"V_{node_id}_{phase}"].strip().split("/")
                    mag = float(mag_str)
                    ang_rad = np.radians(float(ang_str))
                    phasors.append(mag * np.exp(1j * ang_rad))

                u_arr[idx]["id"] = node_id
                u_arr[idx]["v"] = np.array(phasors, dtype=np.complex128)

            # 2. Build Branch Current Array (i)
            i_arr = np.empty(len(branch_ids), dtype=i_dtype)
            for idx, branch_name in enumerate(branch_ids):
                phasors = []
                for phase in ("a", "b", "c"):
                    mag_str, ang_str = row[f"I_{branch_name}_{phase}"].strip().split("/")
                    mag = float(mag_str)
                    ang_rad = np.radians(float(ang_str))
                    phasors.append(mag * np.exp(1j * ang_rad))

                branch_id = 5 if branch_name == "12" else 6
                i_arr[idx]["id"] = branch_id
                i_arr[idx]["i"] = np.array(phasors, dtype=np.complex128)

            results[conn] = {"u": u_arr, "i": i_arr}

    return results


def create_nodes(u_primary: float, u_secondary: float, secondary_3_wire: bool) -> np.ndarray:
    num_nodes = 4
    ids = [1, 2, 3, 4]
    u = [u_primary, u_primary, u_secondary, u_secondary]
    if secondary_3_wire:
        num_nodes = 5
        ids += [10]
        u += [u_secondary]

    node = initialize_array(DatasetType.input, ComponentType.node, num_nodes)
    node[AttributeType.id] = ids
    node[AttributeType.u_rated] = u

    return node


def create_asym_lines(
    primary_3_wire: bool = True,
    secondary_3_wire: bool = False,
) -> np.ndarray:
    asym_line = initialize_array(DatasetType.input, ComponentType.asym_line, 2)
    asym_line[AttributeType.id] = [5, 6]
    asym_line[AttributeType.from_node] = [1, 3]
    asym_line[AttributeType.to_node] = [2, 4]
    asym_line[AttributeType.from_status] = [1, 1]
    asym_line[AttributeType.to_status] = [1, 1]

    length_1_feet = 2000
    length_2_feet = 2500

    wire_params = load_wire_params_dict(WIRE_PARAM_FILE)

    cfg1 = wire_params["3-wire"] if primary_3_wire else wire_params["4-wire"]
    cfg2 = wire_params["3-wire"] if secondary_3_wire else wire_params["4-wire"]

    for attr in ["aa", "ba", "bb", "ca", "cb", "cc"]:
        asym_line[getattr(AttributeType, f"r_{attr}")] = [
            cfg1[f"r_{attr}"] * length_1_feet / MILES_TO_FEET,
            cfg2[f"r_{attr}"] * length_2_feet / MILES_TO_FEET,
        ]
        asym_line[getattr(AttributeType, f"x_{attr}")] = [
            cfg1[f"x_{attr}"] * length_1_feet / MILES_TO_FEET,
            cfg2[f"x_{attr}"] * length_2_feet / MILES_TO_FEET,
        ]

    for attr in ["na", "nb", "nc", "nn"]:
        asym_line[getattr(AttributeType, f"r_{attr}")] = [np.nan, np.nan]
        asym_line[getattr(AttributeType, f"x_{attr}")] = [np.nan, np.nan]

    asym_line[AttributeType.c0] = [1.0e-40, 1.0e-40]
    asym_line[AttributeType.c1] = [1.0e-40, 1.0e-40]

    return asym_line


def create_transformer(
    u1: float, u2: float, winding_from: WindingType, winding_to: WindingType, clock: int
) -> np.ndarray:
    if winding_to != WindingType.delta:
        transformer = initialize_array(DatasetType.input, ComponentType.transformer, 1)
        transformer[AttributeType.id] = [7]
        transformer[AttributeType.from_node] = [2]
        transformer[AttributeType.to_node] = [3]
        transformer[AttributeType.u1] = [u1]
        transformer[AttributeType.uk] = [np.sqrt(np.square(6.0e-2) + np.square(1.0e-2))]
        transformer[AttributeType.pk] = [6e6 * 0.01]
        transformer[AttributeType.winding_from] = [winding_from]
        transformer[AttributeType.winding_to] = [winding_to]
        transformer[AttributeType.clock] = [clock]
    else:
        transformer = initialize_array(DatasetType.input, ComponentType.transformer, 2)
        transformer[AttributeType.id] = [7, 11]
        transformer[AttributeType.from_node] = [2, 4]
        transformer[AttributeType.to_node] = [3, 10]
        transformer[AttributeType.u1] = [u1, u2]
        transformer[AttributeType.uk] = [np.sqrt(np.square(6.0e-2) + np.square(1.0e-2)), 1e-5]
        transformer[AttributeType.pk] = [6e6 * 0.01, 0.0]
        transformer[AttributeType.winding_from] = [winding_from, WindingType.delta]
        transformer[AttributeType.winding_to] = [winding_to, WindingType.wye_n]
        transformer[AttributeType.clock] = [clock, 11]

    transformer[AttributeType.from_status] = [1]
    transformer[AttributeType.to_status] = [1]
    transformer[AttributeType.u2] = [u2]
    transformer[AttributeType.sn] = [6e6]
    transformer[AttributeType.i0] = [0.0]
    transformer[AttributeType.p0] = [0.0]
    transformer[AttributeType.tap_side] = [0]
    transformer[AttributeType.tap_pos] = [1]
    transformer[AttributeType.tap_nom] = [1]
    transformer[AttributeType.tap_min] = [1]
    transformer[AttributeType.tap_max] = [1]
    transformer[AttributeType.tap_size] = [100]
    return transformer


def create_source() -> np.ndarray:
    source = initialize_array(DatasetType.input, ComponentType.source, 1)
    source[AttributeType.id] = [8]
    source[AttributeType.node] = [1]
    source[AttributeType.status] = [1]
    source[AttributeType.u_ref] = [1.0]
    source[AttributeType.sk] = [1.0e50]
    return source


def create_asym_load(p_phases: list[float], pf_phases: list[float], secondary_3_wire: bool) -> np.ndarray:
    q_specified = [p * np.tan(np.arccos(pf)) for p, pf in zip(p_phases, pf_phases)]

    asym_load = initialize_array(DatasetType.input, ComponentType.asym_load, 1)
    asym_load[AttributeType.id] = [9]
    asym_load[AttributeType.node] = [10 if secondary_3_wire else 4]
    asym_load[AttributeType.status] = [1]
    asym_load[AttributeType.type] = [LoadGenType.const_power]
    asym_load[AttributeType.p_specified] = p_phases
    asym_load[AttributeType.q_specified] = q_specified
    return asym_load


def generate_scenarios() -> dict[str, ScenarioConfig]:
    scenarios: dict[str, ScenarioConfig] = {}

    for trafo_type in ["step_down", "step_up"]:
        for load_type in ["balanced", "unbalanced"]:
            expected_results = load_expected_results(curr_directory / f"results_{trafo_type}_{load_type}_load.csv")
            for trafo_connection in ["ynyn", "ynd", "yd", "dyn", "dd"]:
                scenario_name = f"{trafo_type}_{trafo_connection}_{load_type}_load"
                scenario_results = expected_results[trafo_connection]
                u_secondary = 4.16e3 if trafo_type == "step_down" else 24.9e3
                winding_from = winding_from_lookup[trafo_connection]
                winding_to = winding_to_lookup[trafo_connection]
                if trafo_connection in ("ynyn", "dd"):
                    clock = 0
                elif trafo_type == "step_down":
                    clock = 1
                else:
                    clock = 11
                load_p = p_bal if load_type == "balanced" else p_un_bal
                load_pf = pf_bal if load_type == "balanced" else pf_un_bal
                scenarios[scenario_name] = {
                    "u_primary": 12.47e3,
                    "u_secondary": u_secondary,
                    "winding_from": winding_from,
                    "winding_to": winding_to,
                    "clock": clock,
                    "p_phases": load_p,
                    "pf_phases": load_pf,
                    "expected_results": scenario_results,
                }
    return scenarios


def generate_scenario_data(scenario: ScenarioConfig) -> Dataset:
    primary_3_wire = scenario["winding_from"] == WindingType.delta
    secondary_3_wire = scenario["winding_to"] == WindingType.delta

    return {
        ComponentType.node: create_nodes(scenario["u_primary"], scenario["u_secondary"], secondary_3_wire),
        ComponentType.asym_line: create_asym_lines(primary_3_wire, secondary_3_wire),
        ComponentType.transformer: create_transformer(
            scenario["u_primary"],
            scenario["u_secondary"],
            scenario["winding_from"],
            scenario["winding_to"],
            scenario["clock"],
        ),
        ComponentType.source: create_source(),
        ComponentType.asym_load: create_asym_load(scenario["p_phases"], scenario["pf_phases"], secondary_3_wire),
    }


def prepare_results(asym_output: Dataset) -> tuple[np.ndarray, np.ndarray]:
    nodes = asym_output["node"]
    node_ids = nodes["id"]

    v_ln = nodes["u"] * np.exp(1j * nodes["u_angle"])

    v_ll = np.column_stack(
        [
            v_ln[:, 0] - v_ln[:, 1],  # va - vb
            v_ln[:, 1] - v_ln[:, 2],  # vb - vc
            v_ln[:, 2] - v_ln[:, 0],  # vc - va
        ]
    )

    node_dtype = np.dtype(
        [
            ("id", node_ids.dtype),
            ("v_ln", "(3,)complex128"),
            ("v_ll", "(3,)complex128"),
        ]
    )

    u_results = np.empty(len(node_ids), dtype=node_dtype)
    u_results["id"] = node_ids
    u_results["v_ln"] = v_ln
    u_results["v_ll"] = v_ll

    # -------------------------------------------------------------------------
    # 2. Complex Branch/Line Currents
    # -------------------------------------------------------------------------
    lines = asym_output["asym_line"]
    line_ids = lines["id"]

    # Convert degrees to radians for 'i_from' branch currents
    s_complex = lines["p_from"] + 1j * lines["q_from"]
    u_complex = v_ln[[0, 2]]
    i_complex_phasors = np.conjugate(s_complex / u_complex)

    # Define NumPy structured datatype for branch currents
    line_dtype = np.dtype(
        [
            ("id", line_ids.dtype),
            ("i", "(3,)complex128"),
        ]
    )

    i_results = np.empty(len(line_ids), dtype=line_dtype)
    i_results["id"] = line_ids
    i_results["i"] = i_complex_phasors

    return u_results, i_results


def verify_results(
    v_complex: np.ndarray,
    i_complex: np.ndarray,
    expected_results: dict[str, np.ndarray],
    node_winding_types: dict[int, WindingType],
) -> tuple[bool, dict[str, float]]:
    """Calculates difference/errors between calculated and expected results.

    Checks max errors against global threshold constants and prints appropriate
    failure messages.

    Parameters:
        v_complex: Structured np.ndarray (fields: 'id', 'v_ln', 'v_ll')
        i_complex: Structured np.ndarray (fields: 'id', 'i')
        expected_results: Dict containing 'u' and 'i' structured arrays
        node_winding_types: Map of node_id -> WindingType (wye=v_ln, delta=v_ll)

    Returns:
        is_valid (bool): True if all max errors <= global tolerances, else False
        max_errors (dict): Dictionary containing max errors found across network
    """
    max_u_mag_err = 0.0
    max_u_angle_err = 0.0
    max_i_mag_err = 0.0
    max_i_angle_err = 0.0

    expected_u = expected_results["u"]  # fields: 'id', 'v'
    expected_i = expected_results["i"]  # fields: 'id', 'i'

    # -------------------------------------------------------------------------
    # 1. Voltage Errors (u)
    # -------------------------------------------------------------------------
    for exp_row in expected_u:
        node_id = exp_row["id"]
        calc_mask = v_complex["id"] == node_id
        if not np.any(calc_mask):
            continue

        calc_row = v_complex[calc_mask][0]

        # Select L-N or L-L based on node winding type
        winding_type = node_winding_types.get(node_id, WindingType.wye)
        is_line_to_line = winding_type == WindingType.delta
        v_calc = calc_row["v_ll"] if is_line_to_line else calc_row["v_ln"]

        v_exp = exp_row["v"]

        actual_mag, actual_deg = np.abs(v_calc), np.angle(v_calc, deg=True)
        expected_mag, expected_deg = np.abs(v_exp), np.angle(v_exp, deg=True)

        mag_diff = np.abs(actual_mag - expected_mag)
        angle_diff = np.abs((actual_deg - expected_deg + 180) % 360 - 180)

        max_u_mag_err = max(max_u_mag_err, float(np.max(mag_diff)))
        max_u_angle_err = max(max_u_angle_err, float(np.max(angle_diff)))

    # -------------------------------------------------------------------------
    # 2. Current Errors (i)
    # -------------------------------------------------------------------------
    for exp_row in expected_i:
        branch_id = exp_row["id"]
        calc_mask = i_complex["id"] == branch_id
        if not np.any(calc_mask):
            continue

        calc_row = i_complex[calc_mask][0]
        i_calc = calc_row["i"]
        i_exp = exp_row["i"]

        actual_mag, actual_deg = np.abs(i_calc), np.angle(i_calc, deg=True)
        expected_mag, expected_deg = np.abs(i_exp), np.angle(i_exp, deg=True)

        mag_diff = np.abs(actual_mag - expected_mag)
        angle_diff = np.abs((actual_deg - expected_deg + 180) % 360 - 180)

        max_i_mag_err = max(max_i_mag_err, float(np.max(mag_diff)))
        max_i_angle_err = max(max_i_angle_err, float(np.max(angle_diff)))

    # Compile max error record
    max_errors = {
        "max_u_mag_err": max_u_mag_err,
        "max_u_angle_err": max_u_angle_err,
        "max_i_mag_err": max_i_mag_err,
        "max_i_angle_err": max_i_angle_err,
    }

    # -------------------------------------------------------------------------
    # 3. Validation Against Global Standard Thresholds
    # -------------------------------------------------------------------------
    failed_checks = []

    if max_u_mag_err > MAX_U_MAG_TOL:
        failed_checks.append(f"Voltage Mag Error ({max_u_mag_err:.4f}) exceeded max tolerance ({MAX_U_MAG_TOL})")
    if max_u_angle_err > MAX_U_ANGLE_TOL:
        failed_checks.append(
            f"Voltage Angle Error ({max_u_angle_err:.4f}°) exceeded max tolerance ({MAX_U_ANGLE_TOL}°)"
        )
    if max_i_mag_err > MAX_I_MAG_TOL:
        failed_checks.append(f"Current Mag Error ({max_i_mag_err:.4f}) exceeded max tolerance ({MAX_I_MAG_TOL})")
    if max_i_angle_err > MAX_I_ANGLE_TOL:
        failed_checks.append(
            f"Current Angle Error ({max_i_angle_err:.4f}°) exceeded max tolerance ({MAX_I_ANGLE_TOL}°)"
        )

    is_valid = len(failed_checks) == 0

    if not is_valid:
        print("\n❌ Verification FAILED — Max error tolerances exceeded:")
        for failure_msg in failed_checks:
            print(f"  • {failure_msg}")

    return is_valid, max_errors


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = generate_scenarios()

    for scenario_name, cfg in scenarios.items():
        print(f"\n--- Running Scenario: {scenario_name} ---")

        input_data = generate_scenario_data(cfg)

        assert_valid_input_data(input_data=input_data, calculation_type=CalculationType.power_flow)
        assert_valid_input_data(input_data=input_data, calculation_type=CalculationType.power_flow, symmetric=False)

        model = PowerGridModel(input_data)

        try:
            sym_output = model.calculate_power_flow(
                symmetric=True,
                error_tolerance=1e-8,
                max_iterations=200,
                calculation_method=CalculationMethod.newton_raphson,
            )

            asym_output = model.calculate_power_flow(
                symmetric=False,
                error_tolerance=1e-8,
                max_iterations=200,
                calculation_method=CalculationMethod.newton_raphson,
            )

            v_complex, i_complex = prepare_results(asym_output)
            node_windings = {
                1: cfg["winding_from"],
                2: cfg["winding_from"],
                3: cfg["winding_to"],
                4: cfg["winding_to"],
            }

            passed, max_errors = verify_results(
                v_complex=v_complex,
                i_complex=i_complex,
                expected_results=cfg["expected_results"],
                node_winding_types=node_windings,
            )
        except PowerGridError as e:
            print(e)
            passed = False

        if passed:
            print(f"✅ Verification passed for {scenario_name}. Saving dataset...")
            print(f"Max errors: {max_errors}")
            scenario_dir = DATA_DIR / scenario_name
            scenario_dir.mkdir(parents=True, exist_ok=True)

            json_serialize_to_file(scenario_dir / "input.json", input_data, DatasetType.input)
            json_serialize_to_file(scenario_dir / "sym_output.json", sym_output, DatasetType.sym_output)
            json_serialize_to_file(scenario_dir / "asym_output.json", asym_output, DatasetType.asym_output)
            for file_type in ["input", "sym_output", "asym_output"]:
                license_path = scenario_dir / f"{file_type}.json.license"
                if not license_path.exists():
                    license_path.write_text(license_content, encoding="utf-8")
        else:
            print(f"⚠️ Verification failed for {scenario_name}. Files were NOT saved.")


if __name__ == "__main__":
    main()
