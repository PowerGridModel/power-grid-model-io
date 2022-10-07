import pytest

from power_grid_model_io.converters.pgm_converter import PgmConverter


@pytest.fixture
def converter():
    converter = PgmConverter()
    return converter


def test_converter__parse_data(converter: PgmConverter):
    input_data = {
        "node":
            [
                {
                    "id": 1,
                    "u_rated": 400.0
                },
                {
                    "id": 2,
                    "u_rated": 400.0
                },
            ]
    }
    converter._parse_data(data=input_data, data_type="input")
