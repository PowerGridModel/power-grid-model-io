# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def simple_network_data():
    """Create simple test network data with nodes, line, source, and load."""
    return {
        "node": np.array(
            [(1, 10500.0), (2, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
        "line": np.array(
            [(3, 1, 2, 1, 1, 0.11, 0.12, 4.1e-05, 0.107, 510.0)],
            dtype=[
                ("id", "i4"),
                ("from_node", "i4"),
                ("to_node", "i4"),
                ("from_status", "i1"),
                ("to_status", "i1"),
                ("r1", "f8"),
                ("x1", "f8"),
                ("c1", "f8"),
                ("tan1", "f8"),
                ("i_n", "f8"),
            ],
        ),
        "source": np.array(
            [(4, 1, 1, 1.0)],
            dtype=[("id", "i4"), ("node", "i4"), ("status", "i1"), ("u_ref", "f8")],
        ),
        "sym_load": np.array(
            [(5, 2, 1, 4)],
            dtype=[("id", "i4"), ("node", "i4"), ("status", "i1"), ("type", "i4")],
        ),
    }


@pytest.fixture
def network_with_transformer():
    """Create network data with transformer."""
    return {
        "node": np.array(
            [(1, 110000.0), (2, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
        "transformer": np.array(
            [(3, 1, 2, 1, 1, 110000.0, 10500.0, 40e6, 0.0, 0.1, 0.0, 5e-6, 0, 1, 1, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
            dtype=[
                ("id", "i4"),
                ("from_node", "i4"),
                ("to_node", "i4"),
                ("from_status", "i1"),
                ("to_status", "i1"),
                ("u1", "f8"),
                ("u2", "f8"),
                ("sn", "f8"),
                ("uk", "f8"),
                ("pk", "f8"),
                ("i0", "f8"),
                ("p0", "f8"),
                ("winding_from", "i4"),
                ("winding_to", "i4"),
                ("clock", "i4"),
                ("tap_side", "i4"),
                ("tap_pos", "i4"),
                ("tap_min", "i4"),
                ("tap_max", "i4"),
                ("tap_nom", "i4"),
                ("tap_size", "f8"),
                ("uk_min", "f8"),
                ("uk_max", "f8"),
                ("pk_min", "f8"),
                ("pk_max", "f8"),
                ("r_grounding_from", "f8"),
                ("x_grounding_from", "f8"),
                ("r_grounding_to", "f8"),
                ("x_grounding_to", "f8"),
            ],
        ),
        "source": np.array(
            [(4, 1, 1, 1.0)],
            dtype=[("id", "i4"), ("node", "i4"), ("status", "i1"), ("u_ref", "f8")],
        ),
    }


@pytest.fixture
def mock_matplotlib():
    """Mock matplotlib to avoid GUI/display dependencies in tests."""
    with patch("power_grid_model_io.utils.print_network.plt") as mock_plt, patch(
        "power_grid_model_io.utils.print_network.mpatches"
    ) as mock_patches, patch("power_grid_model_io.utils.print_network.PdfPages") as mock_pdf:
        # Create mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        yield {
            "plt": mock_plt,
            "mpatches": mock_patches,
            "PdfPages": mock_pdf,
            "fig": mock_fig,
            "ax": mock_ax,
        }


def test_import():
    """Test that print_network can be imported."""
    from power_grid_model_io.utils.print_network import print_network

    assert callable(print_network)


def test_import_from_submodule():
    """Test that print_network can be imported from submodule."""
    from power_grid_model_io.utils.print_network import NetworkDiagramRenderer, print_network

    assert callable(print_network)
    assert NetworkDiagramRenderer is not None


def test_print_network_empty_data_raises_error(mock_matplotlib):
    """Test that empty network data raises ValueError."""
    from power_grid_model_io.utils.print_network import print_network

    with pytest.raises(ValueError, match="Network data is empty"):
        print_network({}, "output.pdf")


def test_print_network_no_nodes_raises_error(mock_matplotlib):
    """Test that network without nodes raises ValueError."""
    from power_grid_model_io.utils.print_network import print_network

    network = {"line": np.array([], dtype=[("id", "i4")])}

    with pytest.raises(ValueError, match="Network must contain at least one node"):
        print_network(network, "output.pdf")


def test_print_network_simple(mock_matplotlib, simple_network_data, tmp_path):
    """Test print_network with simple network data."""
    from power_grid_model_io.utils.print_network import print_network

    output_file = tmp_path / "test_output.pdf"

    # This should not raise any exceptions
    print_network(simple_network_data, str(output_file))

    # Verify that rendering methods were called
    mock_matplotlib["plt"].subplots.assert_called_once()
    mock_matplotlib["ax"].set_aspect.assert_called_once_with("equal")
    mock_matplotlib["ax"].axis.assert_called_once_with("off")


def test_print_network_adds_pdf_extension(mock_matplotlib, simple_network_data, tmp_path):
    """Test that .pdf extension is added if not present."""
    from power_grid_model_io.utils.print_network import print_network

    output_file = tmp_path / "test_output"  # No extension

    print_network(simple_network_data, str(output_file))

    # Verify PdfPages was called with .pdf extension
    mock_matplotlib["PdfPages"].assert_called_once()
    called_path = mock_matplotlib["PdfPages"].call_args[0][0]
    assert called_path.endswith(".pdf")


def test_print_network_with_transformer(mock_matplotlib, network_with_transformer, tmp_path):
    """Test print_network with transformer component."""
    from power_grid_model_io.utils.print_network import print_network

    output_file = tmp_path / "transformer_output.pdf"

    # This should not raise any exceptions
    print_network(network_with_transformer, str(output_file))

    # Verify basic rendering was performed
    assert mock_matplotlib["plt"].subplots.called
    assert mock_matplotlib["ax"].set_aspect.called


def test_network_diagram_renderer_parse_nodes(simple_network_data):
    """Test that NetworkDiagramRenderer correctly parses nodes."""
    from power_grid_model_io.utils.print_network import NetworkDiagramRenderer

    renderer = NetworkDiagramRenderer(simple_network_data)
    renderer._parse_network_data()

    assert len(renderer.graph.nodes) == 2
    assert 1 in renderer.graph.nodes
    assert 2 in renderer.graph.nodes


def test_network_diagram_renderer_parse_lines(simple_network_data):
    """Test that NetworkDiagramRenderer correctly parses lines."""
    from power_grid_model_io.utils.print_network import NetworkDiagramRenderer

    renderer = NetworkDiagramRenderer(simple_network_data)
    renderer._parse_network_data()

    assert len(renderer.graph.edges) == 1
    assert (1, 2) in renderer.graph.edges
    assert renderer.graph.edges[1, 2]["type"] == "line"


def test_network_diagram_renderer_parse_transformers(network_with_transformer):
    """Test that NetworkDiagramRenderer correctly parses transformers."""
    from power_grid_model_io.utils.print_network import NetworkDiagramRenderer

    renderer = NetworkDiagramRenderer(network_with_transformer)
    renderer._parse_network_data()

    assert len(renderer.graph.edges) == 1
    assert (1, 2) in renderer.graph.edges
    assert renderer.graph.edges[1, 2]["type"] == "transformer"


def test_network_diagram_renderer_calculate_layout(simple_network_data):
    """Test that layout calculation produces positions for all nodes."""
    from power_grid_model_io.utils.print_network import NetworkDiagramRenderer

    renderer = NetworkDiagramRenderer(simple_network_data)
    renderer._parse_network_data()
    renderer._calculate_layout()

    # Check that all nodes have positions
    assert len(renderer.positions) == len(renderer.graph.nodes)
    assert all(isinstance(pos, tuple) and len(pos) == 2 for pos in renderer.positions.values())


def test_network_diagram_renderer_with_all_components():
    """Test NetworkDiagramRenderer with various component types."""
    from power_grid_model_io.utils.print_network import NetworkDiagramRenderer

    network_data = {
        "node": np.array(
            [(1, 10500.0), (2, 10500.0), (3, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
        "line": np.array(
            [(10, 1, 2, 1, 1, 0.11, 0.12, 4.1e-05, 0.107, 510.0)],
            dtype=[
                ("id", "i4"),
                ("from_node", "i4"),
                ("to_node", "i4"),
                ("from_status", "i1"),
                ("to_status", "i1"),
                ("r1", "f8"),
                ("x1", "f8"),
                ("c1", "f8"),
                ("tan1", "f8"),
                ("i_n", "f8"),
            ],
        ),
        "source": np.array(
            [(20, 1, 1, 1.0)],
            dtype=[("id", "i4"), ("node", "i4"), ("status", "i1"), ("u_ref", "f8")],
        ),
        "sym_load": np.array(
            [(30, 2, 1, 4)],
            dtype=[("id", "i4"), ("node", "i4"), ("status", "i1"), ("type", "i4")],
        ),
        "sym_gen": np.array(
            [(40, 3, 1, 4)],
            dtype=[("id", "i4"), ("node", "i4"), ("status", "i1"), ("type", "i4")],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    renderer._parse_network_data()

    # Verify all nodes are parsed
    assert len(renderer.graph.nodes) == 3

    # Verify edges are parsed
    assert len(renderer.graph.edges) == 1


def test_network_diagram_renderer_empty_network():
    """Test NetworkDiagramRenderer with empty network (no edges)."""
    from power_grid_model_io.utils.print_network import NetworkDiagramRenderer

    network_data = {
        "node": np.array(
            [(1, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    renderer._parse_network_data()
    renderer._calculate_layout()

    # Should handle single node gracefully
    assert len(renderer.graph.nodes) == 1
    assert len(renderer.positions) == 1


@patch("power_grid_model_io.utils.print_network.plt")
@patch("power_grid_model_io.utils.print_network.PdfPages")
def test_print_network_closes_figure(mock_pdf, mock_plt, simple_network_data, tmp_path):
    """Test that matplotlib figure is properly closed after saving."""
    from power_grid_model_io.utils.print_network import print_network

    output_file = tmp_path / "test.pdf"

    # Setup mocks
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    print_network(simple_network_data, str(output_file))

    # Verify figure was closed
    mock_plt.close.assert_called_once_with(mock_fig)


def test_network_with_three_winding_transformer():
    """Test network with three-winding transformer."""
    from power_grid_model_io.utils.print_network import NetworkDiagramRenderer

    network_data = {
        "node": np.array(
            [(1, 110000.0), (2, 10500.0), (3, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
        "three_winding_transformer": np.array(
            [(100, 1, 2, 3, 1, 1, 1, 110000.0, 10500.0, 10500.0, 40e6, 40e6, 40e6, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 5e-6, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
            dtype=[
                ("id", "i4"),
                ("node_1", "i4"),
                ("node_2", "i4"),
                ("node_3", "i4"),
                ("status_1", "i1"),
                ("status_2", "i1"),
                ("status_3", "i1"),
                ("u1", "f8"),
                ("u2", "f8"),
                ("u3", "f8"),
                ("sn_1", "f8"),
                ("sn_2", "f8"),
                ("sn_3", "f8"),
                ("uk_12", "f8"),
                ("uk_13", "f8"),
                ("uk_23", "f8"),
                ("pk_12", "f8"),
                ("pk_13", "f8"),
                ("pk_23", "f8"),
                ("i0", "f8"),
                ("p0", "f8"),
                ("winding_1", "i4"),
                ("winding_2", "i4"),
                ("winding_3", "i4"),
                ("clock_12", "i4"),
                ("clock_13", "i4"),
                ("tap_side", "i4"),
                ("tap_pos", "i4"),
                ("tap_min", "i4"),
                ("tap_max", "i4"),
                ("tap_nom", "i4"),
                ("tap_size", "f8"),
                ("uk_12_min", "f8"),
                ("uk_12_max", "f8"),
                ("uk_13_min", "f8"),
                ("uk_13_max", "f8"),
                ("uk_23_min", "f8"),
                ("uk_23_max", "f8"),
                ("pk_12_min", "f8"),
                ("pk_12_max", "f8"),
                ("pk_13_min", "f8"),
                ("pk_13_max", "f8"),
                ("pk_23_min", "f8"),
                ("pk_23_max", "f8"),
            ],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    renderer._parse_network_data()

    # Should create auxiliary node and edges for three-winding transformer
    assert len(renderer.graph.nodes) > 3  # 3 original + 1 auxiliary


def test_network_with_link():
    """Test network with link (bus-to-bus connection)."""
    from power_grid_model_io.utils.print_network import NetworkDiagramRenderer

    network_data = {
        "node": np.array(
            [(1, 10500.0), (2, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
        "link": np.array(
            [(10, 1, 2, 1, 1)],
            dtype=[
                ("id", "i4"),
                ("from_node", "i4"),
                ("to_node", "i4"),
                ("from_status", "i1"),
                ("to_status", "i1"),
            ],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    renderer._parse_network_data()

    assert len(renderer.graph.edges) == 1
    assert (1, 2) in renderer.graph.edges
    assert renderer.graph.edges[1, 2]["type"] == "link"
