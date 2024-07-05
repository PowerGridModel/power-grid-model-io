# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
This module provides the ExcelAmbiguityChecker class, which is designed to identify and report ambiguous column names
within the sheets of an Excel (.xlsx) file. It parses the Excel file, extracts the names of columns from a specified
row across all sheets, and checks for any duplicates within those names to flag them as ambiguous.

Usage:
    checker = ExcelAmbiguityChecker(file_path='path/to/excel/file.xlsx', column_name_in_row=0)
    has_ambiguity, ambiguous_columns = checker.check_ambiguity()
    if has_ambiguity:
        print("Ambiguous column names found:", ambiguous_columns)
    else:
        print("No ambiguous column names found.")

Requirements:
    - Python 3.9 or higher (PGM library dependencies)
    - xml.etree.ElementTree for parsing XML structures within the Excel file.
    - zipfile to handle the Excel file as a ZIP archive for parsing.
"""
import os
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from typing import Dict, List, Optional, Tuple

XML_NAME_SPACE = {"": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}  # NOSONAR
WORK_BOOK = "xl/workbook.xml"
SHARED_STR_PATH = "xl/sharedStrings.xml"
FIND_T = ".//t"
FIND_C = ".//c"
FIND_V = ".//v"
NAME = "name"
FIND_ROW = ".//row"
FIND_SHEET = ".//sheet"
FIND_TYPE = "t"
TYPE_STR = "s"


class ExcelAmbiguityChecker:
    """
    A class to check for ambiguous column names within the sheets of an Excel (.xlsx) file.

    Attributes:
        _file_path (str): The path to the Excel file to be checked.
        _col_name_in_row (int): The row index (0-based) where column names are expected. Default is 0.
        sheets (dict): A dictionary storing sheet names as keys and lists of column names as values.

    Methods:
        __init__(self, file_path, column_name_in_row=0): Initializes the ExcelAmbiguityChecker instance.
        _parse_zip(self, zip_file): Parses the shared strings XML file within the Excel ZIP archive.
        _get_column_names_from_row(self, row, shared_strings): Extracts column names from a specified row.
        _parse_excel_file(self): Parses the Excel file to extract sheet names and their corresponding column names.
    """

    def __init__(self, file_path, column_name_in_row=0) -> None:
        """
        Initializes the ExcelAmbiguityChecker with the path to an Excel file and the row index for column names.

        Parameters:
            file_path (str): The path to the Excel file.
            column_name_in_row (int): The row index (0-based) where column names are expected. Default is 0.
        """
        self._valid_file = file_path.endswith(".xlsx") and os.path.exists(file_path)
        if self._valid_file:
            self._file_path = file_path
            self._col_name_in_row = column_name_in_row
            self.sheets: Dict[str, List[str]] = {}
            self._parse_excel_file()

    def _parse_zip(self, zip_file) -> List[Optional[str]]:
        """
        Parses the shared strings XML file within the Excel ZIP archive to extract all shared strings.

        Parameters:
            zip_file (zipfile.ZipFile): The opened Excel ZIP file.

        Returns:
            list: A list of shared strings used in the Excel file.
        """
        shared_strings_path = SHARED_STR_PATH
        shared_strings = []
        with zip_file.open(shared_strings_path) as f:
            tree = ET.parse(f)
            for si in tree.findall(FIND_T, namespaces=XML_NAME_SPACE):
                shared_strings.append(si.text)
        return shared_strings

    def _get_column_names_from_row(self, row, shared_strings) -> List[Optional[str]]:
        """
        Extracts column names from a specified row using shared strings for strings stored in the shared string table.

        Parameters:
            row (xml.etree.ElementTree.Element): The XML element representing the row.
            shared_strings (list): A list of shared strings extracted from the Excel file.

        Returns:
            list: A list of column names found in the row.
        """
        column_names = []
        for c in row.findall(FIND_C, namespaces=XML_NAME_SPACE):
            cell_type = c.get(FIND_TYPE)
            value = c.find(FIND_V, namespaces=XML_NAME_SPACE)
            if cell_type == TYPE_STR and value is not None:
                column_names.append(shared_strings[int(value.text)])
            elif value is not None:
                column_names.append(value.text)
            else:
                column_names.append(None)
        return column_names

    def _parse_excel_file(self) -> None:
        """
        Parses the Excel file to extract sheet names and their corresponding column names.
        """
        with zipfile.ZipFile(self._file_path) as z:
            shared_strings = self._parse_zip(z)
            workbook_xml = z.read(WORK_BOOK)
            xml_tree = ET.fromstring(workbook_xml)
            sheets = xml_tree.findall(FIND_SHEET, namespaces=XML_NAME_SPACE)

            for index, sheet in enumerate(sheets, start=1):
                sheet_name = str(sheet.get(NAME))
                sheet_file_path = f"xl/worksheets/sheet{index}.xml"

                with z.open(sheet_file_path) as f:
                    sheet_tree = ET.parse(f)
                    rows = sheet_tree.findall(FIND_ROW, namespaces=XML_NAME_SPACE)
                    if rows:
                        column_names = self._get_column_names_from_row(rows[self._col_name_in_row], shared_strings)
                        self.sheets[sheet_name] = [name for name in column_names if name is not None]

    def list_sheets(self) -> List[str]:
        """
        Get the list of all sheet names in the Excel file.

        Returns:
            List[str]: list of all sheet names
        """
        return list(self.sheets.keys())

    def check_ambiguity(self) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Check if there is ambiguity in column names across sheets.

        Returns:
            Tuple[bool, Dict[str, List[str]]]: A tuple containing a boolean indicating if any ambiguity was found,
            and a dictionary with sheet names as keys and lists of ambiguous column names as values.
        """
        res: Dict[str, List[str]] = {}
        if not self._valid_file:
            return False, res
        for sheet_name, column_names in self.sheets.items():
            column_name_counts = Counter(column_names)
            duplicates = [name for name, count in column_name_counts.items() if count > 1]
            if duplicates:
                res[sheet_name] = duplicates
        return bool(res), res


# Example usage
if __name__ == "__main__":
    excel_file_checker = ExcelAmbiguityChecker("excel_ambiguity_check_data.xlsx")
    excel_file_checker.check_ambiguity()
