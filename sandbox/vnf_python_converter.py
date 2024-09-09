from pathlib import Path
from typing import List
import re

import pandas as pd

from power_grid_model_io.converters.tabular_converter import TabularConverter
from power_grid_model_io.data_types.tabular_data import TabularData

from power_grid_model import PowerGridModel
from power_grid_model.utils import json_serialize

file_path = Path(__file__).parent 
vnf = file_path / "data/vision_validation.vnf"
mapping = file_path / "vnf_mapping.yaml"

def vnv_to_pgm_json(vnf: str|Path) -> dict:
    """
    This function will mimmick the vnf to pgm json conversion in pgm-io-native.
    Input vnf converter in pgm-io-native: 
    - .vnf file path

    Output vnf converter in pgm-io-native:
    - PGM json file with input data (no update, no extra info)
    """
    component_dict = parse_vnf(vnf)
    component_data = TabularData(NODE=component_dict["NODE"], LINE=component_dict["LINE"], SOURCE=component_dict["SOURCE"], LOAD=component_dict["LOAD"], CABLE=component_dict["CABLE"])
    converter = TabularConverter(mapping_file=mapping)
    input_data, extra_info = converter.load_input_data(data=component_data)
    # since pgm-io-native returns a json file, we have to do serialization here, even though we already have input_data
    serialized_data = json_serialize(input_data)
    return serialized_data



def parse_vnf(file_path: str|Path) -> pd.DataFrame:
    with open(file_path, "r") as file:
        file = file.read()
        components = ["NODE", "LINE", "SOURCE", "LOAD", "CABLE"]
        component_dict = {}
        for component in components:
            component_dict[component] = parse_component(component, file)
    return component_dict 


def parse_component(component: str, file: str) -> pd.DataFrame:
        # Define the regular expression pattern
        pattern = rf"\[{component}\](.*?)\[\]"

        # Find all matches in the file content
        matches = re.findall(pattern, file, re.DOTALL)

        assert len(matches) in [0,1], f"Found {len(matches)} matches for component {component}."

        if len(matches) == 1:
            match component:
                case "NODE":
                    component_df = parse_node(matches[0])
                case "LINE":
                    component_df = parse_line(matches[0])
                case "SOURCE":
                    component_df = parse_source(matches[0])
                case "LOAD":
                    component_df = parse_load(matches[0])
                case "CABLE":
                    component_df = parse_cable(matches[0])
                case _:
                    raise NotImplementedError(f"Component {component} not implemented.")
        return component_df


def parse_node(vnf_txt: str) -> pd.DataFrame:
    nodes_list = []
    lines = vnf_txt.strip().split("\n")
    for line in lines:
        if line.startswith('#General'):
            guid = re.search(r"GUID:'{(.*?)}'", line).group(1)
            unom = re.search(r"Unom:(\d+(\.\d+)?)", line).group(1)
            nodes_list.append([guid, unom])
    nodes_df = pd.DataFrame(nodes_list, columns=["GUID", "Unom"])
    return nodes_df


def parse_line(vnf_txt: str) -> pd.DataFrame:
    # TODO: parse cables as well
    line_general = []
    line_part = []
    lines = vnf_txt.strip().split("\n")
    for line in lines:
        if line.startswith('#General'):
            guid = re.search(r"GUID:'{(.*?)}'", line).group(1)
            node1 = re.search(r"Node1:'{(.*?)}'", line).group(1)
            node2 = re.search(r"Node2:'{(.*?)}'", line).group(1)
            switch_state1 = re.search(r"SwitchState1:(\d)", line).group(1)
            switch_state2 = re.search(r"SwitchState2:(\d)", line).group(1)
            line_general.append([guid, node1, node2, switch_state1, switch_state2])
        if line.startswith('#LinePart'):
            # TODO check if a line can have multiple line parts
            r = re.search(r"R:(\d+(\.\d+)?)", line).group(1)
            x = re.search(r"X:(\d+(\.\d+)?)", line).group(1)
            c = re.search(r"C:(\d+(\.\d+)?)", line).group(1)
            r0 = re.search(r"R0:(\d+(\.\d+)?)", line).group(1)
            x0 = re.search(r"X0:(\d+(\.\d+)?)", line).group(1)
            c0 = re.search(r"C0:(\d+(\.\d+)?)", line).group(1)
            # TODO check with p2p which 1nom is needed 1/2/3
            inom1 = re.search(r"Inom1:(\d+(\.\d+)?)", line).group(1)
            inom2 = re.search(r"Inom2:(\d+(\.\d+)?)", line).group(1)
            inom3 = re.search(r"Inom3:(\d+(\.\d+)?)", line).group(1)
            tr = re.search(r"TR:(\d+(\.\d+)?)", line).group(1)
            length = re.search(r"Length:(\d+(\.\d+)?)", line).group(1)
            line_part.append([r, x, c, r0, x0, c0, inom1, inom2, inom3, tr, length])
    assert len(line_general) == len(line_part), "Number of line general and line part do not match."
    lines_list = [line_general[i] + line_part[i] for i in range(len(line_general))]
    lines_df = pd.DataFrame(lines_list, columns=["GUID", "Node1", "Node2", "SwitchState1", "SwitchState2", "R", "X", "C", "R0", "X0", "C0", "Inom1", "Inom2", "Inom3", "TR", "Length"])
    return lines_df


def parse_cable(vnf_txt: str) -> pd.DataFrame:
    cable_general = []
    cable_part = []
    lines = vnf_txt.strip().split("\n")
    for line in lines:
        if line.startswith('#General'):
            guid = re.search(r"GUID:'{(.*?)}'", line).group(1)
            node1 = re.search(r"Node1:'{(.*?)}'", line).group(1)
            node2 = re.search(r"Node2:'{(.*?)}'", line).group(1)
            switch_state1 = re.search(r"SwitchState1:(\d)", line).group(1)
            switch_state2 = re.search(r"SwitchState2:(\d)", line).group(1)
            cable_general.append([guid, node1, node2, switch_state1, switch_state2])
        if line.startswith('#CableType'):
            # TODO check if a cable can have multiple cable parts
            r = re.search(r"R:(\d+(\.\d+)?)", line).group(1)
            x = re.search(r"X:(\d+(\.\d+)?)", line).group(1)
            c = re.search(r"C:(\d+(\.\d+)?)", line).group(1)
            r0 = re.search(r"R0:(\d+(\.\d+)?)", line).group(1)
            x0 = re.search(r"X0:(\d+(\.\d+)?)", line).group(1)
            c0 = re.search(r"C0:(\d+(\.\d+)?)", line).group(1)
            # TODO check with p2p which 1nom is needed 0/1/2/3
            inom0 = re.search(r"Inom0:(\d+(\.\d+)?)", line).group(1)
            inom1 = re.search(r"Inom1:(\d+(\.\d+)?)", line).group(1)
            inom2 = re.search(r"Inom2:(\d+(\.\d+)?)", line).group(1)
            inom3 = re.search(r"Inom3:(\d+(\.\d+)?)", line).group(1)
            cable_part.append([r, x, c, r0, x0, c0, inom0, inom1, inom2, inom3])
    assert len(cable_general) == len(cable_part), "Number of line general and line part do not match."
    cable_list = [cable_general[i] + cable_part[i] for i in range(len(cable_general))]
    cable_df = pd.DataFrame(cable_list, columns=["GUID", "Node1", "Node2", "SwitchState1", "SwitchState2", "R", "X", "C", "R0", "X0", "C0", "Inom0", "Inom1", "Inom2", "Inom3"])
    return cable_df


def parse_source(vnf_txt: str) -> pd.DataFrame:
    sources_list = []
    lines = vnf_txt.strip().split("\n")
    for line in lines:
        if line.startswith('#General'):
            guid = re.search(r"GUID:'{(.*?)}'", line).group(1)
            node = re.search(r"Node:'{(.*?)}'", line).group(1)
            switch_state = re.search(r"SwitchState:(\d)", line).group(1)
            uref = re.search(r"Uref:(\d+(\.\d+)?)", line).group(1)
            sknom = re.search(r"Sk2nom:(\d+(\.\d+)?)", line).group(1)
            rx_ratio = re.search(r"R/X:(\d+(\.\d+)?)", line).group(1)
            z01_ratio = re.search(r"Z0/Z1:(\d+(\.\d+)?)", line).group(1)
            sources_list.append([guid, node, switch_state, uref, sknom, rx_ratio, z01_ratio])
    sources_df = pd.DataFrame(sources_list, columns=["GUID", "Node", "SwitchState", "Uref", "Sk2nom", "R/X", "Z0/Z1"])
    return sources_df


def parse_load(vnf_txt: str) -> pd.DataFrame:
    loads_list = []
    lines = vnf_txt.strip().split("\n")
    for line in lines:
        if line.startswith('#General'):
            guid = re.search(r"GUID:'{(.*?)}'", line).group(1)
            node = re.search(r"Node:'{(.*?)}'", line).group(1)
            switch_state = re.search(r"SwitchState:(\d)", line).group(1)
            p = re.search(r"P:(\d+(\.\d+)?)", line).group(1)
            q = re.search(r"Q:(\d+(\.\d+)?)", line).group(1)
            loads_list.append([guid, node, switch_state, p, q])
    loads_df = pd.DataFrame(loads_list, columns=["GUID", "Node", "SwitchState", "P", "Q"])
    # TODO: how to include "gelijktijdigheid / simultanity factor"? See excel converter yaml -> can be in NODE
    # TODO: how to inlcude "load behaviour"? See excel converter yaml
    # TODO: how to include the right multiplier?
    return loads_df


def parse_link(vnf_txt: str) -> pd.DataFrame:
    links_list = []
    lines = vnf_txt.strip().split("\n")
    for line in lines:
        if line.startswith('#General'):
            guid = re.search(r"GUID:'{(.*?)}'", line).group(1)
            node1 = re.search(r"Node1:'{(.*?)}'", line).group(1)
            node2 = re.search(r"Node2:'{(.*?)}'", line).group(1)
            switch_state1 = re.search(r"SwitchState1:(\d)", line).group(1)
            switch_state2 = re.search(r"SwitchState2:(\d)", line).group(1)
            links_list.append([guid, node1, node2, switch_state1, switch_state2])
    links_df = pd.DataFrame(links_list, columns=["GUID", "Node1", "Node2", "SwitchState1", "SwitchState2"])
    return links_df


component_dict = parse_vnf(vnf)
for comp, df in component_dict.items():
    print("===", comp, "===")
    print(df)
    print()

component_data = TabularData(NODE=component_dict["NODE"], LINE=component_dict["LINE"], SOURCE=component_dict["SOURCE"], LOAD=component_dict["LOAD"], CABLE=component_dict["CABLE"])
converter = TabularConverter(mapping_file=mapping)
input_data, extra_info = converter.load_input_data(data=component_data)
print(input_data)

model = PowerGridModel(input_data)
result = model.calculate_power_flow()
print("======== Result ========")
print(result['node'])
print(result['line'])



