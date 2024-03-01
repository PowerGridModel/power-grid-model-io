# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""
    This is a stand-alone conversion script that converts the GUID based Vision 
    excel files to a number based format. The script is designed to be used as 
    a light weight tool to run before running the PGM-IO conversion scripts.
    Example usage:

    new_file = convert_guid_vision_excel("vision_97_en.xlsx", number="Number", {"N1": "Grounding1"})
    nieuw_bestand = convert_guid_vision_excel("vision_97_nl.xlsx", number="Nummer", {"N1": "Arding1"})

"""

import os
import re
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from power_grid_model_io.data_stores.base_data_store import (
    DICT_KEY_NUMBER,
    DICT_KEY_SUBNUMBER,
    LANGUAGE_EN,
    LANGUAGE_NL,
    VISION_EXCEL_LAN_DICT,
)

special_nodes_en = [
    "Transformer loads",
    "Sources",
    "Synchronous generators",
    "Wind turbines",
    "Loads",
    "Capacitors",
    "Reactors",
    "Zigzag transformers",
    "Pvs",
]
special_nodes_nl = [
    "Transformatorbelastingen",
    "Netvoedingen",
    "Synchrone generatoren",
    "Windturbines",
    "Belastingen",
    "Condensatoren",
    "Spoelen",
    "Nulpuntstransformatoren",
    "Pv's",
]


class UUID2IntCvtr:
    """
    Class for bookkeeping the conversion of GUIDs to integers
    """

    def __init__(self, uuids: Optional[list] = None) -> None:
        """Initialize with a list of UUIDs, or empty list

        Args:
            uuids (list, optional): GUID's in Vision excel. Defaults to [].
        """
        if uuids is None:
            uuids = []
        self._uuids_int: dict[str, int] = {}
        self._counter: int = 0
        for uuid in set(uuids):
            self.add(uuid)

    def add_list(self, uuids: list) -> None:
        """Add a list of UUIDs

        Args:
            uuids (list): GUID's in Vision excel
        """
        for uuid in set(uuids):
            self.add(uuid)

    def add(self, uuid: str) -> None:
        """Add a single entry of UUID

        Args:
            uuid (str): A single GUID in Vision excel
        """
        if uuid not in self._uuids_int and str(uuid).lower() != "nan":
            self._uuids_int[uuid] = self._counter
            self._counter += 1

    def query(self, uuid: str) -> Optional[int]:
        """Get the singular integer value respective of a UUID input

        Args:
            uuid (str): A single GUID in Vision excel

        Returns:
            str: the singular integer value
        """
        try:
            return self._uuids_int[uuid]
        except KeyError:
            return None

    def get_keys(self) -> list:
        """Get the keys in the dictionary

        Returns:
            list: the whole list of keys
        """
        return list(self._uuids_int.keys())

    def get_size(self) -> int:
        """Get how many valid UUIDs are in the dictionary

        Returns:
            int: the size of the dictionary
        """
        return self._counter


def load_excel_file(file_name: Union[Path, str]) -> pd.ExcelFile:
    """Load an excel file

    Args:
        file_name (str): Excel file name

    Returns:
        pd.ExcelFile: pandas ExcelFile object
    """
    return pd.ExcelFile(file_name)


def get_guid_columns(df: pd.DataFrame) -> list:
    """Get the columns that contain the word "GUID"

    Args:
        df (pd.DataFrame): panda dataframe

    Returns:
        list: list of columns containing the word "GUID"
    """
    guid_regex = re.compile(r".*GUID$")
    return df.filter(regex=guid_regex).columns


def add_guid_values_to_cvtr(df: pd.DataFrame, guid_column: str, cvtr: UUID2IntCvtr) -> None:
    """Add the GUID values to the UUID2IntCvtr object

    Args:
        df (pd.DataFrame): panda dataframe
        guid_column (str): column name containing the word "GUID"
        cvtr (UUID2IntCvtr): the UUID2IntCvtr object
    """
    cvtr.add_list(df[guid_column].tolist())


def get_special_key_map(sheet_name: str, nodes_en: list[str], nodes_nl: list[str]) -> dict:
    """Get the special nodes for English and Dutch

    Args:
        sheet_name (str): the sheet name
        mapping (dict): the mapping dictionary
    """
    if sheet_name in nodes_en:
        return VISION_EXCEL_LAN_DICT[LANGUAGE_EN]
    if sheet_name in nodes_nl:
        return VISION_EXCEL_LAN_DICT[LANGUAGE_NL]
    return {}


def insert_or_update_number_column(
    df: pd.DataFrame, guid_column: str, sheet_name: str, cvtr: UUID2IntCvtr, number: str
) -> None:
    """Insert the number column or update the number column if it already exists

    Args:
        df (pd.DataFrame): panda dataframe
        guid_column (str): column name containing the substring "GUID"
        cvtr (UUID2IntCvtr): the UUID2IntCvtr object
        number (str): "Number" or "Nummer" depending on the language
    """
    new_column_name = guid_column.replace("GUID", number)
    special_key_mapping = get_special_key_map(sheet_name, special_nodes_en, special_nodes_nl)

    if guid_column == "GUID" and special_key_mapping not in (None, {}):
        new_column_name = guid_column.replace("GUID", special_key_mapping[DICT_KEY_SUBNUMBER])
    try:
        df.insert(df.columns.get_loc(guid_column) + 1, new_column_name, df[guid_column].apply(cvtr.query))
    except ValueError:
        df[new_column_name] = df[guid_column].apply(cvtr.query)


def update_column_names(df: pd.DataFrame, terms_changed: dict) -> None:
    """Update column names according to user input dictionary

    Args:
        df (pd.DataFrame): Pandas dataframe
        terms_changed (dict): the dictionary containing the terms to be changed
    """
    df.rename(columns=terms_changed, inplace=True)


def save_df_to_excel(df: pd.DataFrame, file_name: str, sheet_name: str, i: int) -> None:
    """Dump the panda dataframe to an excel file

    Args:
        df (pd.DataFrame): panda dataframe
        file_name (str): file name
        sheet_name (str): the sheet name to write
        i (int): counter
    """
    if i == 0:
        df.to_excel(file_name, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(file_name, engine="openpyxl", mode="a") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def convert_guid_vision_excel(
    excel_file: Union[Path, str],
    number: str = VISION_EXCEL_LAN_DICT[LANGUAGE_EN][DICT_KEY_NUMBER],
    terms_changed: Optional[dict] = None,
) -> str:
    """Main entry function. Convert the GUID based Vision excel files to a number based format

    Args:
        excel_file (Path | str): Vision excel file name
        number (str): "Number" or "Nummer" depending on the language. Defaults to "Number".
        terms_changed (dict): the dictionary containing the terms to be changed. Defaults to {}.

    Returns:
        str: the new excel file name
    """
    if terms_changed is None:
        terms_changed = {}
    xls = load_excel_file(excel_file)
    cvtr = UUID2IntCvtr()
    dir_name, file_name = os.path.split(excel_file)
    new_excel_name = os.path.join(dir_name, f"new_{file_name}")

    for i, sheet_name in enumerate(xls.sheet_names):
        df = xls.parse(sheet_name)
        update_column_names(df, terms_changed)
        guid_columns = get_guid_columns(df)

        for guid_column in guid_columns:
            add_guid_values_to_cvtr(df, guid_column, cvtr)
            insert_or_update_number_column(df, guid_column, sheet_name, cvtr, number)

        save_df_to_excel(df, new_excel_name, sheet_name, i)

    return new_excel_name
