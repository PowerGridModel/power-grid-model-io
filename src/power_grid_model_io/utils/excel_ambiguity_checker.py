# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import xml.etree.ElementTree as ET
import zipfile
from collections import Counter

XML_NAME_SPACE = {"": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


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
        self._file_path = file_path
        self._col_name_in_row = column_name_in_row
        self.sheets = {}
        self._parse_excel_file()

    def _parse_zip(self, zip_file) -> list:
        """
        Parses the shared strings XML file within the Excel ZIP archive to extract all shared strings.

        Parameters:
            zip_file (zipfile.ZipFile): The opened Excel ZIP file.

        Returns:
            list: A list of shared strings used in the Excel file.
        """
        shared_strings_path = "xl/sharedStrings.xml"
        shared_strings = []
        with zip_file.open(shared_strings_path) as f:
            tree = ET.parse(f)
            for si in tree.findall(".//t", namespaces=XML_NAME_SPACE):
                shared_strings.append(si.text)
        return shared_strings

    def _get_column_names_from_row(self, row, shared_strings) -> list:
        """
        Extracts column names from a specified row using shared strings for strings stored in the shared string table.

        Parameters:
            row (xml.etree.ElementTree.Element): The XML element representing the row.
            shared_strings (list): A list of shared strings extracted from the Excel file.

        Returns:
            list: A list of column names found in the row.
        """
        column_names = []
        for c in row.findall(".//c", namespaces=XML_NAME_SPACE):
            cell_type = c.get("t")
            value = c.find(".//v", namespaces=XML_NAME_SPACE)
            if cell_type == "s" and value is not None:
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
            workbook_xml = z.read("xl/workbook.xml")
            xml_tree = ET.fromstring(workbook_xml)
            sheets = xml_tree.findall(".//sheet", namespaces=XML_NAME_SPACE)

            for index, sheet in enumerate(sheets, start=1):
                sheet_name = sheet.get("name")
                sheet_file_path = f"xl/worksheets/sheet{index}.xml"

                with z.open(sheet_file_path) as f:
                    sheet_tree = ET.parse(f)
                    rows = sheet_tree.findall(".//row", namespaces=XML_NAME_SPACE)
                    if rows:
                        column_names = self._get_column_names_from_row(rows[self._col_name_in_row], shared_strings)
                        self.sheets[sheet_name] = column_names

    def check_ambiguity(self) -> bool:
        """
        Check if there is ambiguity in column names across sheets.

        Returns:
            bool: result
        """
        res = False
        for sheet_name, column_names in self.sheets.items():
            column_name_counts = Counter(column_names)
            duplicates = [name for name, count in column_name_counts.items() if count > 1]
            if duplicates:
                print(f"In sheet: {sheet_name}, ambiguious column names: {duplicates}\n")
                res = True
        return res


# Example usage
if __name__ == "__main__":
    excel_file_checker = ExcelAmbiguityChecker("data.xlsx")
    excel_file_checker.check_ambiguity()
