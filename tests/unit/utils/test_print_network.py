# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from power_grid_model_io.utils.print_network import NetworkDiagramRenderer, print_network


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
            [
                (
                    3,
                    1,
                    2,
                    1,
                    1,
                    110000.0,
                    10500.0,
                    40e6,
                    0.0,
                    0.1,
                    0.0,
                    5e-6,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            ],
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
    with (
        patch("power_grid_model_io.utils.print_network.plt") as mock_plt,
        patch("power_grid_model_io.utils.print_network.mpatches") as mock_patches,
        patch("power_grid_model_io.utils.print_network.PdfPages") as mock_pdf,
    ):
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


@pytest.mark.usefixtures("mock_matplotlib")
def test_print_network_empty_data_raises_error():
    """Test that empty network data raises ValueError."""

    with pytest.raises(ValueError, match="Network data is empty"):
        print_network({}, "output.pdf")


@pytest.mark.usefixtures("mock_matplotlib")
def test_print_network_no_nodes_raises_error():
    """Test that network without nodes raises ValueError."""

    network = {"line": np.array([], dtype=[("id", "i4")])}

    with pytest.raises(ValueError, match="Network must contain at least one node"):
        print_network(network, "output.pdf")


def test_print_network_simple(mock_matplotlib, simple_network_data, tmp_path):
    """Test print_network with simple network data."""

    output_file = tmp_path / "test_output.pdf"

    # This should not raise any exceptions
    print_network(simple_network_data, str(output_file))

    # Verify that rendering methods were called
    mock_matplotlib["plt"].subplots.assert_called_once()
    mock_matplotlib["ax"].set_aspect.assert_called_once_with("equal")
    mock_matplotlib["ax"].axis.assert_called_once_with("off")


def test_print_network_adds_pdf_extension(mock_matplotlib, simple_network_data, tmp_path):
    """Test that .pdf extension is added if not present."""

    output_file = tmp_path / "test_output"  # No extension

    print_network(simple_network_data, str(output_file))

    # Verify PdfPages was called with .pdf extension
    mock_matplotlib["PdfPages"].assert_called_once()
    called_path = mock_matplotlib["PdfPages"].call_args[0][0]
    assert called_path.endswith(".pdf")


def test_print_network_with_transformer(mock_matplotlib, network_with_transformer, tmp_path):
    """Test print_network with transformer component."""

    output_file = tmp_path / "transformer_output.pdf"

    # This should not raise any exceptions
    print_network(network_with_transformer, str(output_file))

    # Verify basic rendering was performed
    assert mock_matplotlib["plt"].subplots.called
    assert mock_matplotlib["ax"].set_aspect.called


def test_network_diagram_renderer_parse_nodes(simple_network_data):
    """Test that NetworkDiagramRenderer correctly parses nodes."""

    renderer = NetworkDiagramRenderer(simple_network_data)
    renderer._parse_network_data()

    assert len(renderer.graph.nodes) == 2
    assert 1 in renderer.graph.nodes
    assert 2 in renderer.graph.nodes


def test_network_diagram_renderer_parse_lines(simple_network_data):
    """Test that NetworkDiagramRenderer correctly parses lines."""

    renderer = NetworkDiagramRenderer(simple_network_data)
    renderer._parse_network_data()

    assert len(renderer.graph.edges) == 1
    assert (1, 2) in renderer.graph.edges
    assert renderer.graph.edges[1, 2]["type"] == "line"


def test_network_diagram_renderer_parse_transformers(network_with_transformer):
    """Test that NetworkDiagramRenderer correctly parses transformers."""

    renderer = NetworkDiagramRenderer(network_with_transformer)
    renderer._parse_network_data()

    assert len(renderer.graph.edges) == 1
    assert (1, 2) in renderer.graph.edges
    assert renderer.graph.edges[1, 2]["type"] == "transformer"


def test_network_diagram_renderer_calculate_layout(simple_network_data):
    """Test that layout calculation produces positions for all nodes."""

    renderer = NetworkDiagramRenderer(simple_network_data)
    renderer._parse_network_data()
    renderer._calculate_layout()

    # Check that all nodes have positions
    assert len(renderer.positions) == len(renderer.graph.nodes)
    assert all(isinstance(pos, tuple) and len(pos) == 2 for pos in renderer.positions.values())


def test_network_diagram_renderer_with_all_components():
    """Test NetworkDiagramRenderer with various component types."""

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


def test_print_network_closes_figure(mock_matplotlib, simple_network_data, tmp_path):
    """Test that matplotlib figure is properly closed after saving."""

    output_file = tmp_path / "test.pdf"

    print_network(simple_network_data, str(output_file))

    # Verify figure was closed
    mock_matplotlib["plt"].close.assert_called_once_with(mock_matplotlib["fig"])


def test_network_with_three_winding_transformer():
    """Test network with three-winding transformer."""

    network_data = {
        "node": np.array(
            [(1, 110000.0), (2, 10500.0), (3, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
        "three_winding_transformer": np.array(
            [
                (
                    100,
                    1,
                    2,
                    3,
                    1,
                    1,
                    1,
                    110000.0,
                    10500.0,
                    10500.0,
                    40e6,
                    40e6,
                    40e6,
                    0.0,
                    0.0,
                    0.0,
                    0.1,
                    0.1,
                    0.1,
                    0.0,
                    0.0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            ],
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
                ("r_grounding_1", "f8"),
                ("x_grounding_1", "f8"),
                ("r_grounding_2", "f8"),
                ("x_grounding_2", "f8"),
                ("r_grounding_3", "f8"),
                ("x_grounding_3", "f8"),
            ],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    renderer._parse_network_data()

    # Should create auxiliary node and edges for three-winding transformer
    assert len(renderer.graph.nodes) > 3  # 3 original + 1 auxiliary


def test_network_with_link():
    """Test network with link (bus-to-bus connection)."""

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


def test_connection_point_allocation_left():
    """Test connection point allocation with left preference."""

    network_data = {
        "node": np.array(
            [(1, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    offset = renderer._allocate_bus_connection_point(1, "left")
    assert offset < 0  # Should be negative (left side)


def test_connection_point_allocation_right():
    """Test connection point allocation with right preference."""

    network_data = {
        "node": np.array(
            [(1, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    offset = renderer._allocate_bus_connection_point(1, "right")
    assert offset > 0  # Should be positive (right side)


def test_connection_point_allocation_exceeds_max_width():
    """Test connection point allocation when exceeding bus width."""

    network_data = {
        "node": np.array(
            [(1, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    # Allocate many connection points to exceed bus width
    offsets = []
    for _ in range(20):
        offset = renderer._allocate_bus_connection_point(1, "auto")
        offsets.append(offset)

    # At least some should be clamped to max_offset
    max_offset = renderer.node_width / 2 - 0.02
    assert any(abs(offset) == max_offset for offset in offsets)


def test_network_with_asymmetric_load():
    """Test network with asymmetric load."""

    network_data = {
        "node": np.array(
            [(1, 10500.0), (2, 10500.0)],
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
        "asym_load": np.array(
            [(30, 2, 1, 4)],
            dtype=[("id", "i4"), ("node", "i4"), ("status", "i1"), ("type", "i4")],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    renderer._parse_network_data()
    renderer._calculate_layout()
    fig, ax = renderer.render()

    # Verify network was rendered
    assert fig is not None
    assert ax is not None


def test_network_with_asymmetric_generator():
    """Test network with asymmetric generator."""

    network_data = {
        "node": np.array(
            [(1, 10500.0), (2, 10500.0)],
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
        "asym_gen": np.array(
            [(40, 2, 1, 4)],
            dtype=[("id", "i4"), ("node", "i4"), ("status", "i1"), ("type", "i4")],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    renderer._parse_network_data()
    renderer._calculate_layout()
    fig, ax = renderer.render()

    # Verify network was rendered
    assert fig is not None
    assert ax is not None


def test_network_with_shunt():
    """Test network with shunt component."""

    network_data = {
        "node": np.array(
            [(1, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
        "shunt": np.array(
            [(50, 1, 1)],
            dtype=[("id", "i4"), ("node", "i4"), ("status", "i1")],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    renderer._parse_network_data()
    renderer._calculate_layout()
    fig, ax = renderer.render()

    # Verify network was rendered
    assert fig is not None
    assert ax is not None


def test_save_to_pdf_without_rendering():
    """Test that save_to_pdf raises error if render not called."""

    network_data = {
        "node": np.array(
            [(1, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)

    with pytest.raises(RuntimeError, match="Diagram has not been rendered yet"):
        renderer.save_to_pdf("output.pdf")


def test_layout_with_no_source_nodes():
    """Test layout calculation when there are no sources (spring layout fallback)."""

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
    }

    renderer = NetworkDiagramRenderer(network_data)
    renderer._parse_network_data()
    renderer._calculate_layout()

    # Should have positions for all nodes using spring layout
    assert len(renderer.positions) == 3


@patch("power_grid_model_io.utils.print_network.nx.multipartite_layout")
def test_layout_fallback_on_exception(mock_multipartite, simple_network_data):
    """Test that layout falls back to spring layout when multipartite fails."""

    # Make multipartite_layout raise an exception
    mock_multipartite.side_effect = ValueError("Test exception")

    renderer = NetworkDiagramRenderer(simple_network_data)
    renderer._parse_network_data()
    renderer._calculate_layout()

    # Should still have positions (from spring layout fallback)
    assert len(renderer.positions) > 0


def test_orthogonal_path_vertically_aligned():
    """Test orthogonal path when points are vertically aligned."""

    network_data = {
        "node": np.array(
            [(1, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    x_coords, _ = renderer._calculate_orthogonal_path(1.0, 2.0, 1.0, 4.0)

    # Should be a straight vertical line
    assert len(x_coords) == 2
    assert x_coords[0] == x_coords[1]


def test_orthogonal_path_horizontally_aligned():
    """Test orthogonal path when points are horizontally aligned."""

    network_data = {
        "node": np.array(
            [(1, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    _, y_coords = renderer._calculate_orthogonal_path(1.0, 2.0, 3.0, 2.0)

    # Should be a straight horizontal line
    assert len(y_coords) == 2
    assert y_coords[0] == y_coords[1]


def test_draw_components_with_missing_ax():
    """Test that drawing components handles missing ax gracefully."""

    network_data = {
        "node": np.array(
            [(1, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
        "source": np.array(
            [(2, 1, 1, 1.0)],
            dtype=[("id", "i4"), ("node", "i4"), ("status", "i1"), ("u_ref", "f8")],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    renderer._parse_network_data()
    renderer._calculate_layout()

    # Call drawing methods without setting up ax (should not crash)
    renderer._draw_node(1, 0.0, 0.0)
    renderer._draw_source(1)
    renderer._draw_load(1, 10)
    renderer._draw_generator(1, 20)
    renderer._draw_shunt(1, 30)


def test_draw_line_with_missing_positions():
    """Test _draw_line when node positions are missing."""

    network_data = {
        "node": np.array(
            [(1, 10500.0), (2, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    renderer._parse_network_data()
    # Don't calculate layout, so positions will be empty

    # This should not crash
    renderer._draw_line(1, 2, 10)


def test_draw_transformer_with_missing_positions():
    """Test _draw_transformer when node positions are missing."""

    network_data = {
        "node": np.array(
            [(1, 10500.0), (2, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    renderer._parse_network_data()
    # Don't calculate layout

    # This should not crash
    renderer._draw_transformer(1, 2, 100)


def test_snap_to_grid():
    """Test grid snapping functionality."""

    network_data = {
        "node": np.array(
            [(1, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    renderer.grid_spacing = 1.0

    # Test snapping
    x, y = renderer._snap_to_grid(1.3, 2.7)
    assert x == 1.0
    assert y == 3.0

    x, y = renderer._snap_to_grid(2.6, 3.4)
    assert x == 3.0
    assert y == 3.0


def test_draw_source_with_missing_position():
    """Test _draw_source when node position is missing."""

    network_data = {
        "node": np.array(
            [(1, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
        "source": np.array(
            [(2, 1, 1, 1.0)],
            dtype=[("id", "i4"), ("node", "i4"), ("status", "i1"), ("u_ref", "f8")],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    renderer._parse_network_data()
    # Don't calculate layout, so positions will be empty

    # This should not crash
    renderer._draw_source(1)


def test_draw_load_with_missing_position():
    """Test _draw_load when node position is missing."""

    network_data = {
        "node": np.array(
            [(1, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    renderer._parse_network_data()
    # Don't calculate layout

    # This should not crash
    renderer._draw_load(1, 10)


def test_draw_shunt_with_missing_position():
    """Test _draw_shunt when node position is missing."""

    network_data = {
        "node": np.array(
            [(1, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    renderer._parse_network_data()
    # Don't calculate layout

    # This should not crash
    renderer._draw_shunt(1, 30)


def test_draw_generator_with_missing_position():
    """Test _draw_generator when node position is missing."""

    network_data = {
        "node": np.array(
            [(1, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    renderer._parse_network_data()
    # Don't calculate layout

    # This should not crash
    renderer._draw_generator(1, 20)


def test_layout_with_source_but_empty_layers():
    """Test layout when we have sources but layers dict ends up empty."""

    network_data = {
        "node": np.array(
            [(1, 10500.0), (2, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
        "source": np.array(
            [(10, 1, 1, 1.0)],
            dtype=[("id", "i4"), ("node", "i4"), ("status", "i1"), ("u_ref", "f8")],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    renderer._parse_network_data()
    renderer._calculate_layout()

    # Should have positions (will use spring layout as fallback)
    assert len(renderer.positions) == 2


def test_shunt_rendering_complete():
    """Test complete shunt rendering with axes."""

    network_data = {
        "node": np.array(
            [(1, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
        "shunt": np.array(
            [(50, 1, 1)],
            dtype=[("id", "i4"), ("node", "i4"), ("status", "i1")],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    _, ax = renderer.render()

    # Verify that shunt was drawn (check that patches were added)
    assert len(ax.patches) > 0  # Should have at least the node rectangle


def test_asymmetric_load_in_render():
    """Test that asymmetric loads are properly rendered in full render."""

    network_data = {
        "node": np.array(
            [(1, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
        "source": np.array(
            [(10, 1, 1, 1.0)],
            dtype=[("id", "i4"), ("node", "i4"), ("status", "i1"), ("u_ref", "f8")],
        ),
        "asym_load": np.array(
            [(30, 1, 1, 4)],
            dtype=[("id", "i4"), ("node", "i4"), ("status", "i1"), ("type", "i4")],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    fig, ax = renderer.render()

    # Verify rendering completed successfully
    assert fig is not None
    assert ax is not None
    # Should have multiple patches (node + load triangle)
    assert len(ax.patches) >= 2


def test_asymmetric_gen_in_render():
    """Test that asymmetric generators are properly rendered in full render."""

    network_data = {
        "node": np.array(
            [(1, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
        "source": np.array(
            [(10, 1, 1, 1.0)],
            dtype=[("id", "i4"), ("node", "i4"), ("status", "i1"), ("u_ref", "f8")],
        ),
        "asym_gen": np.array(
            [(40, 1, 1, 4)],
            dtype=[("id", "i4"), ("node", "i4"), ("status", "i1"), ("type", "i4")],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    fig, ax = renderer.render()

    # Verify rendering completed successfully
    assert fig is not None
    assert ax is not None
    # Should have multiple patches (node + generator circle)
    assert len(ax.patches) >= 2


def test_calculate_layout_with_empty_graph():
    """Test _calculate_layout when graph has no nodes (early return)."""

    network_data = {}  # Empty network

    renderer = NetworkDiagramRenderer(network_data)
    renderer._parse_network_data()
    renderer._calculate_layout()

    # Should have no positions and not crash
    assert len(renderer.positions) == 0


def test_layout_with_source_not_in_graph():
    """Test layout when source node is not in the graph."""

    network_data = {
        "node": np.array(
            [(1, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
        "source": np.array(
            [(10, 999, 1, 1.0)],  # Source refers to node 999 which doesn't exist
            dtype=[("id", "i4"), ("node", "i4"), ("status", "i1"), ("u_ref", "f8")],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    renderer._parse_network_data()
    renderer._calculate_layout()

    # Should still work with spring layout fallback
    assert len(renderer.positions) == 1


def test_connection_point_with_narrow_node():
    """Test connection point allocation that exceeds bus width."""

    network_data = {
        "node": np.array(
            [(1, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    # Make node very narrow so we quickly exceed max offset
    renderer.node_width = 0.1
    renderer.connection_spacing = 0.1

    # Allocate many points - some should be clamped
    for i in range(10):
        _ = renderer._allocate_bus_connection_point(1, "right" if i % 2 == 0 else "left")

    # The last ones should have been clamped
    max_offset = renderer.node_width / 2 - 0.02
    offsets = renderer.bus_connection_offsets[1]
    # At least one should be at the max
    assert any(abs(offset) == max_offset for offset in offsets)


def test_transformer_drawing_with_zero_distance():
    """Test transformer drawing when circle positions have zero distance."""

    network_data = {
        "node": np.array(
            [(1, 110000.0), (2, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
        "transformer": np.array(
            [
                (
                    3,
                    1,
                    2,
                    1,
                    1,
                    110000.0,
                    10500.0,
                    40e6,
                    0.0,
                    0.1,
                    0.0,
                    5e-6,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            ],
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
    }

    renderer = NetworkDiagramRenderer(network_data)
    fig, ax = renderer.render()

    # Should complete without error
    assert fig is not None
    assert ax is not None


def test_three_winding_transformer_edge_drawing():
    """Test that three-winding transformer edges are drawn."""

    network_data = {
        "node": np.array(
            [(1, 110000.0), (2, 10500.0), (3, 10500.0)],
            dtype=[("id", "i4"), ("u_rated", "f8")],
        ),
        "three_winding_transformer": np.array(
            [
                (
                    100,
                    1,
                    2,
                    3,
                    1,
                    1,
                    1,
                    110000.0,
                    10500.0,
                    10500.0,
                    40e6,
                    40e6,
                    40e6,
                    0.0,
                    0.0,
                    0.0,
                    0.1,
                    0.1,
                    0.1,
                    0.0,
                    0.0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            ],
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
                ("r_grounding_1", "f8"),
                ("x_grounding_1", "f8"),
                ("r_grounding_2", "f8"),
                ("x_grounding_2", "f8"),
                ("r_grounding_3", "f8"),
                ("x_grounding_3", "f8"),
            ],
        ),
    }

    renderer = NetworkDiagramRenderer(network_data)
    fig, ax = renderer.render()

    # Should complete successfully
    assert fig is not None
    assert len(ax.lines) > 0  # Should have drawn transformer branches
