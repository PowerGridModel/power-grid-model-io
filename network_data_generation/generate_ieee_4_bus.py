# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
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
from power_grid_model.errors import PowerGridError
from power_grid_model.utils import json_serialize_to_file
from power_grid_model.validation import assert_valid_input_data

root = Path(__file__).parent.parent
DATA_DIR = root / "src" / "power_grid_model_io" / "networks" / "data" / "ieee" / "ieee_four_bus"

INPUT_FILE = DATA_DIR / "input.json"
SYM_OUTPUT_FILE = DATA_DIR / "sym_output.json"
ASYM_OUTPUT_FILE = DATA_DIR / "asym_output.json"


class ScenarioConfig(TypedDict):
    u_primary: float
    u_secondary: float
    winding_from: WindingType
    winding_to: WindingType
    clock: int
    p_phases: list[float]
    pf_phases: list[float]
    expected_voltages: dict[int, list[tuple[float, float]]]


license_content = (
    "# SPDX-FileCopyrightText: Contributors to the Power Grid Model project \n#\n# SPDX-License-Identifier: MPL-2.0\n"
)

MILES_TO_FEET = 5.280e3

WIRE_PARAMS = {
    "3-wire": {
        "R": {
            "aa": 0.4013,
            "ba": 0.0953,
            "bb": 0.4013,
            "ca": 0.0953,
            "cb": 0.0953,
            "cc": 0.4013,
        },
        "X": {
            "aa": 1.4133,
            "ba": 0.8515,
            "bb": 1.4133,
            "ca": 0.7266,
            "cb": 0.7802,
            "cc": 1.4133,
        },
    },
    "4-wire": {
        "R": {
            "aa": 0.4576,
            "ba": 0.1559,
            "bb": 0.4666,
            "ca": 0.1535,
            "cb": 0.1580,
            "cc": 0.4615,
        },
        "X": {
            "aa": 1.0780,
            "ba": 0.5017,
            "bb": 1.0482,
            "ca": 0.3849,
            "cb": 0.4236,
            "cc": 1.0651,
        },
    },
}


def create_nodes(u_primary: float, u_secondary: float, secondary_3_wire: bool) -> np.ndarray:
    if secondary_3_wire:
        node = initialize_array(DatasetType.input, ComponentType.node, 5)
        node[AttributeType.id] = [1, 2, 3, 4, 10]
        node[AttributeType.u_rated] = [u_primary, u_primary, u_secondary, u_secondary, u_secondary]
    else:
        node = initialize_array(DatasetType.input, ComponentType.node, 4)
        node[AttributeType.id] = [1, 2, 3, 4]
        node[AttributeType.u_rated] = [u_primary, u_primary, u_secondary, u_secondary]
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

    cfg1 = WIRE_PARAMS["3-wire"] if primary_3_wire else WIRE_PARAMS["4-wire"]
    cfg2 = WIRE_PARAMS["3-wire"] if secondary_3_wire else WIRE_PARAMS["4-wire"]

    for phase in ["aa", "ba", "bb", "ca", "cb", "cc"]:
        asym_line[getattr(AttributeType, f"r_{phase}")] = [
            cfg1["R"][phase] * length_1_feet / MILES_TO_FEET,
            cfg2["R"][phase] * length_2_feet / MILES_TO_FEET,
        ]
        asym_line[getattr(AttributeType, f"x_{phase}")] = [
            cfg1["X"][phase] * length_1_feet / MILES_TO_FEET,
            cfg2["X"][phase] * length_2_feet / MILES_TO_FEET,
        ]

    for phase in ["na", "nb", "nc", "nn"]:
        asym_line[getattr(AttributeType, f"r_{phase}")] = [np.nan, np.nan]
        asym_line[getattr(AttributeType, f"x_{phase}")] = [np.nan, np.nan]

    asym_line[AttributeType.c0] = [1.0e-40, 1.0e-40]
    asym_line[AttributeType.c1] = [1.0e-40, 1.0e-40]
    asym_line[AttributeType.i_n] = [2000, 2000]

    return asym_line


def create_transformer(
    u1: float, u2: float, winding_from: WindingType, winding_to: WindingType, clock: int
) -> np.ndarray:
    if winding_to != WindingType.delta:
        transformer = initialize_array(DatasetType.input, ComponentType.transformer, 1)
        transformer[AttributeType.id] = [7]
        transformer[AttributeType.from_node] = [2]
        transformer[AttributeType.to_node] = [3]
        transformer[AttributeType.from_status] = [1]
        transformer[AttributeType.to_status] = [1]
        transformer[AttributeType.u1] = [u1]
        transformer[AttributeType.u2] = [u2]
        transformer[AttributeType.sn] = [6e6]
        transformer[AttributeType.uk] = [np.sqrt(np.square(6.0e-2) + np.square(1.0e-2))]
        transformer[AttributeType.pk] = [6e6 * 0.01]
        transformer[AttributeType.i0] = [0.0]
        transformer[AttributeType.p0] = [0.0]
        transformer[AttributeType.winding_from] = [winding_from]
        transformer[AttributeType.winding_to] = [winding_to]
        transformer[AttributeType.clock] = [clock]
        transformer[AttributeType.tap_side] = [0]
        transformer[AttributeType.tap_pos] = [1]
        transformer[AttributeType.tap_nom] = [1]
        transformer[AttributeType.tap_min] = [1]
        transformer[AttributeType.tap_max] = [1]
        transformer[AttributeType.tap_size] = [100]
    else:
        transformer = initialize_array(DatasetType.input, ComponentType.transformer, 2)
        transformer[AttributeType.id] = [7, 11]
        transformer[AttributeType.from_node] = [2, 4]
        transformer[AttributeType.to_node] = [3, 10]
        transformer[AttributeType.from_status] = [1, 1]
        transformer[AttributeType.to_status] = [1, 1]
        transformer[AttributeType.u1] = [u1, u2]
        transformer[AttributeType.u2] = [u2, u2]
        transformer[AttributeType.sn] = [6e6, 6e6]
        transformer[AttributeType.uk] = [np.sqrt(np.square(6.0e-2) + np.square(1.0e-2)), 1e-5]
        transformer[AttributeType.pk] = [6e6 * 0.01, 0.0]
        transformer[AttributeType.i0] = [0.0, 0.0]
        transformer[AttributeType.p0] = [0.0, 0.0]
        transformer[AttributeType.winding_from] = [winding_from, WindingType.delta]
        transformer[AttributeType.winding_to] = [winding_to, WindingType.wye_n]
        transformer[AttributeType.clock] = [clock, 11]
        transformer[AttributeType.tap_side] = [0, 0]
        transformer[AttributeType.tap_pos] = [1, 1]
        transformer[AttributeType.tap_nom] = [1, 1]
        transformer[AttributeType.tap_min] = [1, 1]
        transformer[AttributeType.tap_max] = [1, 1]
        transformer[AttributeType.tap_size] = [100, 100]
    return transformer


def create_source() -> np.ndarray:
    source = initialize_array(DatasetType.input, ComponentType.source, 1)
    source[AttributeType.id] = [8]
    source[AttributeType.node] = [1]
    source[AttributeType.status] = [1]
    source[AttributeType.u_ref] = [1.0]
    source[AttributeType.sk] = [1.0e40]
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


def generate_scenario(scenario: ScenarioConfig) -> dict[ComponentType, np.ndarray]:
    #     u_primary: float,
    #     u_secondary: float,
    #     winding_from: WindingType,
    #     winding_to: WindingType,
    #     clock: int,
    #     p_phases: list[float],
    #     pf_phases: list[float],
    # ) -> dict[ComponentType, np.ndarray]:

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


def compute_node_complex_voltages(asym_output: dict[str, Any]) -> pd.DataFrame:
    u = pd.DataFrame(asym_output["node"]["u"], columns=["va", "vb", "vc"], index=asym_output["node"]["id"])
    u_angle = pd.DataFrame(
        asym_output["node"]["u_angle"], columns=["va_a", "vb_a", "vc_a"], index=asym_output["node"]["id"]
    )

    v_complex = pd.DataFrame(index=u.index)
    v_complex["va"] = u["va"] * np.exp(1j * u_angle["va_a"])
    v_complex["vb"] = u["vb"] * np.exp(1j * u_angle["vb_a"])
    v_complex["vc"] = u["vc"] * np.exp(1j * u_angle["vc_a"])
    v_complex["vab"] = v_complex["va"] - v_complex["vb"]
    v_complex["vbc"] = v_complex["vb"] - v_complex["vc"]
    v_complex["vca"] = v_complex["vc"] - v_complex["va"]
    return v_complex


def verify_node_voltages(  # noqa: PLR0913, PLR0917
    calculated_complex: pd.DataFrame,
    expected_voltages: dict[int, list[tuple[float, float]]],
    node_winding_types: dict[int, WindingType],
    rtol: float = 1e-3,
    v_atol: float = 1.0,
    angle_atol: float = 0.3,
) -> bool:
    """
    Verifies magnitude and phase angle for specified nodes.

    expected_voltages maps node_id -> list of (magnitude, angle_deg) tuples per phase.
    Example: {4: [(2175.0, -4.1), (2255.0, -123.6), (2203.0, 114.8)]}
    """
    all_passed = True

    for node_idx, expected_phased_data in expected_voltages.items():
        winding_type = node_winding_types.get(node_idx)
        is_line_to_line = winding_type == WindingType.delta

        cols = ["vab", "vbc", "vca"] if is_line_to_line else ["va", "vb", "vc"]

        # Extract complex vectors and compute magnitudes + angles in degrees
        v_complex_node = calculated_complex.loc[node_idx, cols].to_numpy()
        actual_mag = np.abs(v_complex_node)
        actual_deg = np.rad2deg(np.angle(v_complex_node))

        # Extract expected magnitudes and angles
        expected_mag = np.array([exp[0] for exp in expected_phased_data])
        expected_deg = np.array([exp[1] for exp in expected_phased_data])

        mag_diff = np.abs(actual_mag - expected_mag)
        angle_diff = np.abs((actual_deg - expected_deg + 180) % 360 - 180)

        # comparison
        mag_passed_mask = np.isclose(actual_mag, expected_mag, rtol=rtol, atol=v_atol)
        angle_passed_mask = np.isclose(actual_deg, expected_deg, rtol=rtol, atol=angle_atol)

        # Combined check per node
        node_passed = np.all(mag_passed_mask) and np.all(angle_passed_mask)

        if not node_passed:
            all_passed = False
            v_type = "L-L" if is_line_to_line else "L-N"
            print(f"\n❌ Validation FAILED at Node {node_idx} ({v_type}):")

            for phase_idx, phase_name in enumerate(cols):
                print(
                    f"  Phase {phase_name.upper()}: "
                    f"Actual = {actual_mag[phase_idx]:.0f}∠{actual_deg[phase_idx]:.1f}° | "
                    f"Expected = {expected_mag[phase_idx]:.0f}∠{expected_deg[phase_idx]:.1f}° | "
                    f"ΔMag = {mag_diff[phase_idx]:.2f}, ΔAngle = {angle_diff[phase_idx]:.2f}°"
                )

    return all_passed


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    p_bal = [1.8e6, 1.8e6, 1.8e6]
    pf_bal = [0.9, 0.9, 0.9]

    p_un_bal = [1.275e6, 1.8e6, 2.375e6]
    pf_un_bal = [0.85, 0.9, 0.95]

    scenarios: dict[str, ScenarioConfig] = {
        "step_down_dyn1_balanced_load": {
            "u_primary": 12.47e3,
            "u_secondary": 4.16e3,
            "winding_from": WindingType.delta,
            "winding_to": WindingType.wye_n,
            "clock": 1,
            "p_phases": p_bal,
            "pf_phases": pf_bal,
            "expected_voltages": {
                2: [(12340.0, 29.7), (12349.0, -90.4), (12318.0, 149.6)],
                3: [(2249.0, -33.7), (2263.0, -153.4), (2259.0, 86.4)],
                4: [(1920.0, -39.1), (2054.0, -158.3), (1986.0, 80.9)],
            },
        },
        "step_down_dyn1_unbalanced_load": {
            "u_primary": 12.47e3,
            "u_secondary": 4.16e3,
            "winding_from": WindingType.delta,
            "winding_to": WindingType.wye_n,
            "clock": 1,
            "p_phases": p_un_bal,
            "pf_phases": pf_un_bal,
            "expected_voltages": {
                2: [(12350.0, 29.6), (12314.0, -90.4), (12333.0, 149.8)],
                3: [(2290.0, -32.4), (2261.0, -153.8), (2214.0, 85.2)],
                4: [(2157.0, -34.2), (1936.0, -157.0), (1849.0, 73.4)],
            },
        },
        "step_down_ynyn0_balanced_load": {
            "u_primary": 12.47e3,
            "u_secondary": 4.16e3,
            "winding_from": WindingType.wye_n,
            "winding_to": WindingType.wye_n,
            "clock": 0,
            "p_phases": p_bal,
            "pf_phases": pf_bal,
            "expected_voltages": {
                2: [(7107.0, -0.3), (7140.0, -120.3), (7121.0, 119.6)],
                3: [(2247.0, -3.7), (2269.0, -123.5), (2256.0, 116.4)],
                4: [(1918.0, -9.1), (2061.0, -128.3), (1981.0, 110.9)],
            },
        },
        "step_down_ynyn0_unbalanced_load": {
            "u_primary": 12.47e3,
            "u_secondary": 4.16e3,
            "winding_from": WindingType.wye_n,
            "winding_to": WindingType.wye_n,
            "clock": 0,
            "p_phases": p_un_bal,
            "pf_phases": pf_un_bal,
            "expected_voltages": {
                2: [(7164.0, -0.1), (7110.0, -120.2), (7082.0, 119.3)],
                3: [(2305.0, -2.3), (2255.0, -123.6), (2203.0, 114.8)],
                4: [(2175.0, -4.1), (1930.0, -126.8), (1833.0, 102.8)],
            },
        },
        "step_down_yd1_balanced_load": {
            "u_primary": 12.47e3,
            "u_secondary": 4.16e3,
            "winding_from": WindingType.wye,
            "winding_to": WindingType.delta,
            "clock": 1,
            "p_phases": p_bal,
            "pf_phases": pf_bal,
            "expected_voltages": {
                2: [(7112.0, -0.03), (7133.0, -120.4), (7124.0, 119.6)],
                3: [(3906.0, -3.4), (3915.0, -123.6), (3909.0, 116.3)],
                4: [(3437.0, -7.8), (3497.0, -129.3), (3388.0, 110.6)],
            },
        },
        "step_down_yd1_unbalanced_load": {
            "u_primary": 12.47e3,
            "u_secondary": 4.16e3,
            "winding_from": WindingType.wye,
            "winding_to": WindingType.delta,
            "clock": 1,
            "p_phases": p_un_bal,
            "pf_phases": pf_un_bal,
            "expected_voltages": {
                2: [(7112.0, -0.2), (7144.0, -120.4), (7112.0, 119.5)],
                3: [(3896.0, -2.8), (3972.0, -123.8), (3874.0, 115.7)],
                4: [(3425.0, -5.8), (3646.0, -130.3), (3298.0, 108.6)],
            },
        },
        "step_down_dd0_balanced_load": {
            "u_primary": 12.47e3,
            "u_secondary": 4.16e3,
            "winding_from": WindingType.delta,
            "winding_to": WindingType.delta,
            "clock": 0,
            "p_phases": p_bal,
            "pf_phases": pf_bal,
            "expected_voltages": {
                2: [(12339.0, 29.7), (12349.0, -90.4), (12321.0, 149.6)],
                3: [(3911.0, 26.5), (3914.0, -93.6), (3905.0, 146.4)],
                4: [(3442.0, 22.3), (3497.0, -99.4), (3384.0, 140.7)],
            },
        },
        "step_down_dd0_unbalanced_load": {
            "u_primary": 12.47e3,
            "u_secondary": 4.16e3,
            "winding_from": WindingType.delta,
            "winding_to": WindingType.delta,
            "clock": 0,
            "p_phases": p_un_bal,
            "pf_phases": pf_un_bal,
            "expected_voltages": {
                2: [(12341.0, 29.8), (12370.0, -90.5), (12302.0, 149.5)],
                3: [(3902.0, 27.2), (3972.0, -93.9), (3871.0, 145.7)],
                4: [(3431.0, 24.3), (3647.0, -100.4), (3294.0, 138.6)],
            },
        },
        "step_up_dyn1_balanced_load": {
            "u_primary": 12.47e3,
            "u_secondary": 24.9e3,
            "winding_from": WindingType.delta,
            "winding_to": WindingType.wye_n,
            "clock": 11,
            "p_phases": p_bal,
            "pf_phases": pf_bal,
            "expected_voltages": {
                2: [(12361.0, 29.7), (12372.0, -90.4), (12348.0, 149.6)],
                3: [(13697.0, 26.7), (13710.0, -93.4), (13681.0, 146.6)],
                4: [(13653.0, 26.6), (13678.0, -93.5), (13644.0, 146.5)],
            },
        },
        "step_up_dyn1_unbalanced_load": {
            "u_primary": 12.47e3,
            "u_secondary": 24.9e3,
            "winding_from": WindingType.delta,
            "winding_to": WindingType.wye_n,
            "clock": 11,
            "p_phases": p_un_bal,
            "pf_phases": pf_un_bal,
            "expected_voltages": {
                2: [(12364.0, 29.8), (12391.0, -90.5), (12333.0, 149.6)],
                3: [(13792.0, 27.7), (13733.0, -93.5), (13641.0, 145.4)],
                4: [(13768.0, 27.7), (13684.0, -93.6), (13600.0, 145.2)],
            },
        },
        "step_up_ynyn0_balanced_load": {
            "u_primary": 12.47e3,
            "u_secondary": 24.9e3,
            "winding_from": WindingType.wye_n,
            "winding_to": WindingType.wye_n,
            "clock": 0,
            "p_phases": p_bal,
            "pf_phases": pf_bal,
            "expected_voltages": {
                2: [(7126.0, -0.3), (7145.0, -120.4), (7137.0, 119.6)],
                3: [(13675.0, -3.3), (13715.0, -123.4), (13698.0, 116.6)],
                4: [(13631.0, -3.5), (13682.0, -123.5), (13661.0, 116.5)],
            },
        },
        "step_up_ynyn0_unbalanced_load": {
            "u_primary": 12.47e3,
            "u_secondary": 24.9e3,
            "winding_from": WindingType.wye_n,
            "winding_to": WindingType.wye_n,
            "clock": 0,
            "p_phases": p_un_bal,
            "pf_phases": pf_un_bal,
            "expected_voltages": {
                2: [(7161.0, -0.1), (7120.0, -120.3), (7128.0, 119.3)],
                3: [(13839.0, -2.1), (13663.0, -123.3), (13655.0, 115.1)],
                4: [(13815.0, -2.2), (13614.0, -123.4), (13615.0, 114.9)],
            },
        },
    }

    for scenario_name, cfg in scenarios.items():
        print(f"\n--- Running Scenario: {scenario_name} ---")

        input_data = generate_scenario(cfg)

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
            v_complex = compute_node_complex_voltages(asym_output)
            node_windings = {
                1: cfg["winding_from"],
                2: cfg["winding_from"],
                3: cfg["winding_to"],
                4: cfg["winding_to"],
            }

            passed = verify_node_voltages(
                calculated_complex=v_complex,
                expected_voltages=cfg["expected_voltages"],
                node_winding_types=node_windings,
                rtol=1e-3,
                v_atol=1.0,
                angle_atol=0.3,
            )
        except PowerGridError as e:
            print(e)
            passed = False

        if passed:
            print(f"✅ Verification passed for {scenario_name}. Saving dataset...")
            json_serialize_to_file(DATA_DIR / f"input_{scenario_name}.json", input_data, DatasetType.input)
            json_serialize_to_file(DATA_DIR / f"sym_output_{scenario_name}.json", sym_output, DatasetType.sym_output)
            json_serialize_to_file(DATA_DIR / f"asym_output_{scenario_name}.json", asym_output, DatasetType.asym_output)
            for file_type in ["input", "sym_output", "asym_output"]:
                license_path = DATA_DIR / f"{file_type}_{scenario_name}.json.license"
                if not license_path.exists():
                    license_path.write_text(license_content, encoding="utf-8")
        else:
            print(f"⚠️ Verification failed for {scenario_name}. Files were NOT saved.")


if __name__ == "__main__":
    main()
