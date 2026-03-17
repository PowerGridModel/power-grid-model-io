# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""
Module for printing Power Grid Model networks to PDF with electrical diagrams.

This module provides functionality to visualize PGM network data as IEEE-standard
electrical diagrams exported to PDF format.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from power_grid_model.data_types import SingleDataset


class NetworkDiagramRenderer:
    """Renders Power Grid Model network data as IEEE-standard electrical diagrams."""

    def __init__(self, network: SingleDataset):
        """
        Initialize the renderer with network data.

        Args:
            network: PGM network data (dict with component types as keys)
        """
        self.network = network
        self.graph = nx.DiGraph()
        self.positions: dict[int, tuple[float, float]] = {}
        self.fig: plt.Figure | None = None
        self.ax: plt.Axes | None = None

        # Symbol sizing parameters
        self.node_width = 0.3
        self.node_height = 0.05
        self.transformer_radius = 0.15
        self.source_radius = 0.12
        self.load_size = 0.15
        self.gen_radius = 0.12

        # Grid parameters for orthogonal routing
        self.grid_spacing = 1.0  # Grid spacing for snapping positions
        self.routing_clearance = 0.3  # Minimum clearance for routing paths

        # Connection point tracking for avoiding overlaps on bus bars
        self.bus_connection_offsets: dict[int, list[float]] = {}  # node_id -> list of x offsets used
        self.connection_spacing = 0.08  # Spacing between connection points on bus

    def _parse_network_data(self) -> None:
        """
        Parse the network data and build a graph structure.

        Extracts nodes and edges from the PGM network data to create
        a NetworkX graph for layout calculations.
        """
        # Add nodes
        if "node" in self.network:
            for node in self.network["node"]:
                node_id = int(node["id"])
                u_rated = float(node["u_rated"]) if "u_rated" in node.dtype.names else 10500.0
                self.graph.add_node(node_id, type="node", u_rated=u_rated)

        # Add lines (branches)
        if "line" in self.network:
            for line in self.network["line"]:
                line_id = int(line["id"])
                from_node = int(line["from_node"])
                to_node = int(line["to_node"])
                self.graph.add_edge(from_node, to_node, type="line", id=line_id)

        # Add transformers
        if "transformer" in self.network:
            for trafo in self.network["transformer"]:
                trafo_id = int(trafo["id"])
                from_node = int(trafo["from_node"])
                to_node = int(trafo["to_node"])
                self.graph.add_edge(from_node, to_node, type="transformer", id=trafo_id)

        # Add three-winding transformers (create auxiliary nodes)
        if "three_winding_transformer" in self.network:
            for trafo3w in self.network["three_winding_transformer"]:
                trafo_id = int(trafo3w["id"])
                node_1 = int(trafo3w["node_1"])
                node_2 = int(trafo3w["node_2"])
                node_3 = int(trafo3w["node_3"])

                # Create auxiliary node for the center of the three-winding transformer
                aux_node_id = trafo_id * 10000  # Use large ID to avoid conflicts
                self.graph.add_node(aux_node_id, type="aux_trafo3w", trafo_id=trafo_id)

                self.graph.add_edge(node_1, aux_node_id, type="trafo3w_branch", id=trafo_id, winding=1)
                self.graph.add_edge(node_2, aux_node_id, type="trafo3w_branch", id=trafo_id, winding=2)
                self.graph.add_edge(node_3, aux_node_id, type="trafo3w_branch", id=trafo_id, winding=3)

        # Add links (bus-to-bus connections)
        if "link" in self.network:
            for link in self.network["link"]:
                link_id = int(link["id"])
                from_node = int(link["from_node"])
                to_node = int(link["to_node"])
                self.graph.add_edge(from_node, to_node, type="link", id=link_id)

    def _snap_to_grid(self, x: float, y: float) -> tuple[float, float]:
        """
        Snap a position to the nearest grid point.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Tuple of (snapped_x, snapped_y)
        """
        snapped_x = round(x / self.grid_spacing) * self.grid_spacing
        snapped_y = round(y / self.grid_spacing) * self.grid_spacing
        return snapped_x, snapped_y

    def _allocate_bus_connection_point(self, node_id: int, preferred_side: str = "auto") -> float:
        """
        Allocate a unique connection point offset on a bus bar.

        Args:
            node_id: The node/bus ID
            preferred_side: "left", "right", or "auto" for automatic allocation

        Returns:
            X offset from bus center for the connection point
        """
        if node_id not in self.bus_connection_offsets:
            self.bus_connection_offsets[node_id] = []

        used_offsets = self.bus_connection_offsets[node_id]

        # Try to allocate on preferred side or find available spot
        if preferred_side == "left":
            # Try negative offsets first
            offset = -self.connection_spacing * (len([o for o in used_offsets if o < 0]) + 1)
        elif preferred_side == "right":
            # Try positive offsets first
            offset = self.connection_spacing * (len([o for o in used_offsets if o > 0]) + 1)
        else:
            # Auto: alternate between left and right
            count = len(used_offsets)
            side = -1 if count % 2 == 0 else 1
            offset = side * self.connection_spacing * ((count + 1) // 2 + 1)

        # Ensure we don't exceed bus width
        max_offset = self.node_width / 2 - 0.02
        if abs(offset) > max_offset:
            # If we run out of space, stack vertically (future enhancement)
            offset = max_offset * (1 if offset > 0 else -1)

        used_offsets.append(offset)
        return offset

    def _calculate_layout(self) -> None:
        """
        Calculate positions for all nodes and components.

        Uses hierarchical layout with sources at the top, then snaps to grid
        for orthogonal routing.
        """
        if len(self.graph.nodes) == 0:
            return

        # Identify sources (nodes with source components attached)
        source_nodes = set()
        if "source" in self.network:
            for source in self.network["source"]:
                source_nodes.add(int(source["node"]))

        # Try hierarchical layout if we have source nodes
        if source_nodes:
            # Create layers based on distance from source
            layers: dict[int, int] = {}
            for source_node in source_nodes:
                if source_node in self.graph:
                    # BFS from each source to assign layers
                    visited = {source_node}
                    queue = [(source_node, 0)]
                    while queue:
                        node, layer = queue.pop(0)
                        if node not in layers or layer < layers[node]:
                            layers[node] = layer
                        for neighbor in self.graph.successors(node):
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append((neighbor, layer + 1))

            # Use multipartite layout if we have layers
            if layers:
                try:
                    self.positions = nx.multipartite_layout(self.graph, subset_key=lambda n: layers.get(n, 0))
                except (ValueError, KeyError, TypeError):
                    # Fall back to spring layout
                    self.positions = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
            else:
                self.positions = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        else:
            # Use spring layout for networks without clear hierarchy
            self.positions = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)

        # Scale positions for better visibility
        scale_factor = max(3.0, len(self.graph.nodes) * 0.5)
        self.positions = {node: (x * scale_factor, y * scale_factor) for node, (x, y) in self.positions.items()}

        # Snap all positions to grid for orthogonal routing
        self.positions = {node: self._snap_to_grid(x, y) for node, (x, y) in self.positions.items()}

    def _draw_node(self, node_id: int, x: float, y: float) -> None:
        """
        Draw a bus/node as a thick horizontal bar (IEEE standard).
        Bus width is dynamically adjusted based on number of connections.

        Args:
            node_id: Node identifier
            x: X coordinate
            y: Y coordinate
        """
        if self.ax is None:
            return

        # Calculate required bus width based on number of connections
        num_connections = len(self.bus_connection_offsets.get(node_id, []))
        min_bus_width = self.node_width
        required_width = max(min_bus_width, num_connections * self.connection_spacing * 1.5)
        bus_width = min(required_width, 1.0)  # Cap at reasonable maximum

        # Draw thick horizontal bar for bus
        bar = mpatches.Rectangle(
            (x - bus_width / 2, y - self.node_height / 2),
            bus_width,
            self.node_height,
            linewidth=2,
            edgecolor="black",
            facecolor="black",
        )
        self.ax.add_patch(bar)

        # Add node ID label
        self.ax.text(x, y + self.node_height + 0.1, f"Node {node_id}", ha="center", va="bottom", fontsize=8)

    def _calculate_orthogonal_path(self, x1: float, y1: float, x2: float, y2: float) -> tuple[list[float], list[float]]:
        """
        Calculate an orthogonal (Manhattan) path between two points.

        Args:
            x1, y1: Start point coordinates
            x2, y2: End point coordinates

        Returns:
            Tuple of (x_coords, y_coords) lists for the path
        """
        # If points are already aligned horizontally or vertically, use direct path
        if abs(x1 - x2) < 0.01:  # Vertically aligned
            return [x1, x2], [y1, y2]
        if abs(y1 - y2) < 0.01:  # Horizontally aligned
            return [x1, x2], [y1, y2]

        # For Manhattan routing, use a two-segment path (L-shape)
        # Choose routing direction based on which creates less conflict
        # Strategy: go horizontal first, then vertical (can be improved with path finding)

        # Calculate midpoint for routing
        mid_x = (x1 + x2) / 2

        # Snap midpoint to grid
        mid_x = round(mid_x / self.grid_spacing) * self.grid_spacing

        # Create L-shaped path: start -> (mid_x, y1) -> (mid_x, y2) -> end
        x_coords = [x1, mid_x, mid_x, x2]
        y_coords = [y1, y1, y2, y2]

        return x_coords, y_coords

    def _draw_line(self, from_node: int, to_node: int, line_id: int) -> None:
        """
        Draw a line connection between two nodes using orthogonal routing with unique bus connection points.

        Args:
            from_node: Source node ID
            to_node: Destination node ID
            line_id: Line identifier
        """
        if self.ax is None or from_node not in self.positions or to_node not in self.positions:
            return

        x1, y1 = self.positions[from_node]
        x2, y2 = self.positions[to_node]

        # Allocate unique connection points on each bus
        offset1 = self._allocate_bus_connection_point(from_node, "auto")
        offset2 = self._allocate_bus_connection_point(to_node, "auto")

        # Adjust starting and ending points to be at the allocated connection points
        x1_conn = x1 + offset1
        x2_conn = x2 + offset2

        # Calculate orthogonal path from connection point to connection point
        x_coords, y_coords = self._calculate_orthogonal_path(x1_conn, y1, x2_conn, y2)

        # Draw orthogonal path
        self.ax.plot(x_coords, y_coords, "k-", linewidth=1.5, zorder=1)

        # Add line ID label at midpoint of path
        mid_idx = len(x_coords) // 2
        mid_x = (x_coords[mid_idx - 1] + x_coords[mid_idx]) / 2
        mid_y = (y_coords[mid_idx - 1] + y_coords[mid_idx]) / 2
        self.ax.text(
            mid_x,
            mid_y,
            f"L{line_id}",
            fontsize=7,
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none"),
        )

    def _draw_transformer(self, from_node: int, to_node: int, trafo_id: int) -> None:
        """
        Draw a two-winding transformer as two 40% overlapping circles (IEEE standard) with orthogonal routing.

        Args:
            from_node: Primary side node ID
            to_node: Secondary side node ID
            trafo_id: Transformer identifier
        """
        if self.ax is None or from_node not in self.positions or to_node not in self.positions:
            return

        x1, y1 = self.positions[from_node]
        x2, y2 = self.positions[to_node]

        # Allocate unique connection points on each bus
        offset1 = self._allocate_bus_connection_point(from_node, "auto")
        offset2 = self._allocate_bus_connection_point(to_node, "auto")

        # Adjust starting and ending points
        x1_conn = x1 + offset1
        x2_conn = x2 + offset2

        # Calculate orthogonal path
        x_coords, y_coords = self._calculate_orthogonal_path(x1_conn, y1, x2_conn, y2)

        # Find the midpoint along the path for transformer symbol placement
        # Calculate total path length
        total_length = 0.0
        for i in range(len(x_coords) - 1):
            dx = x_coords[i + 1] - x_coords[i]
            dy = y_coords[i + 1] - y_coords[i]
            total_length += np.sqrt(dx**2 + dy**2)

        # Find position at half the path length
        target_length = total_length / 2
        current_length = 0.0
        cx1, cy1, cx2, cy2 = x1, y1, x2, y2  # defaults

        for i in range(len(x_coords) - 1):
            dx = x_coords[i + 1] - x_coords[i]
            dy = y_coords[i + 1] - y_coords[i]
            segment_length = np.sqrt(dx**2 + dy**2)

            if current_length + segment_length >= target_length:
                # This segment contains the midpoint
                ratio = (target_length - current_length) / segment_length if segment_length > 0 else 0.5

                # Unit vector for this segment
                ux, uy = (dx / segment_length, dy / segment_length) if segment_length > 0 else (1, 0)

                # Position circles around the midpoint
                mid_x = x_coords[i] + dx * ratio
                mid_y = y_coords[i] + dy * ratio

                offset = self.transformer_radius * 1.5
                cx1 = mid_x - ux * offset
                cy1 = mid_y - uy * offset
                cx2 = mid_x + ux * offset
                cy2 = mid_y + uy * offset
                break

            current_length += segment_length

        # Draw the orthogonal path
        self.ax.plot(x_coords, y_coords, "k-", linewidth=1.5, zorder=1)

        # Draw two 40% overlapping circles (IEEE standard for transformer)
        # Calculate overlap: circles should overlap by 40% of diameter
        overlap_distance = self.transformer_radius * 2 * 0.6  # 60% of diameter = centers distance

        # Recalculate circle positions with overlap
        mid_x = (cx1 + cx2) / 2
        mid_y = (cy1 + cy2) / 2
        dx = cx2 - cx1
        dy = cy2 - cy1
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 0:
            ux, uy = dx / dist, dy / dist
        else:
            ux, uy = 1, 0

        # Position circles with 40% overlap
        cx1 = mid_x - ux * overlap_distance / 2
        cy1 = mid_y - uy * overlap_distance / 2
        cx2 = mid_x + ux * overlap_distance / 2
        cy2 = mid_y + uy * overlap_distance / 2

        # First draw white-filled circles with white outline to clear the area
        circle1_bg = mpatches.Circle(
            (cx1, cy1), self.transformer_radius, edgecolor="white", facecolor="white", linewidth=2, zorder=2
        )
        circle2_bg = mpatches.Circle(
            (cx2, cy2), self.transformer_radius, edgecolor="white", facecolor="white", linewidth=2, zorder=2
        )
        self.ax.add_patch(circle1_bg)
        self.ax.add_patch(circle2_bg)

        # Then draw black outlines without fill so both circles are fully visible
        circle1_outline = mpatches.Circle(
            (cx1, cy1), self.transformer_radius, edgecolor="black", facecolor="none", linewidth=2, zorder=3
        )
        circle2_outline = mpatches.Circle(
            (cx2, cy2), self.transformer_radius, edgecolor="black", facecolor="none", linewidth=2, zorder=3
        )
        self.ax.add_patch(circle1_outline)
        self.ax.add_patch(circle2_outline)

        # Add transformer ID label
        self.ax.text(mid_x, mid_y + self.transformer_radius + 0.1, f"T{trafo_id}", fontsize=7, ha="center", va="bottom")

    def _draw_source(self, node_id: int) -> None:
        """
        Draw a voltage source symbol.

        Args:
            node_id: Node ID where the source is connected
        """
        if self.ax is None or node_id not in self.positions:
            return

        x, y = self.positions[node_id]

        # Draw source above the node
        source_y = y + self.node_height / 2 + self.source_radius + 0.15

        # Draw circle for source
        circle = mpatches.Circle(
            (x, source_y), self.source_radius, edgecolor="black", facecolor="white", linewidth=2, zorder=2
        )
        self.ax.add_patch(circle)

        # Draw sine wave inside (simplified as ~)
        self.ax.text(x, source_y, "~", fontsize=14, ha="center", va="center", weight="bold")

        # Draw connection line to node
        self.ax.plot([x, x], [y + self.node_height / 2, source_y - self.source_radius], "k-", linewidth=1.5, zorder=1)

        # Add label
        self.ax.text(x, source_y + self.source_radius + 0.05, "Source", fontsize=7, ha="center", va="bottom")

    def _draw_load(self, node_id: int, load_id: int) -> None:
        """
        Draw a load as a triangle pointing down (IEEE standard).

        Args:
            node_id: Node ID where the load is connected
            load_id: Load identifier
        """
        if self.ax is None or node_id not in self.positions:
            return

        x, y = self.positions[node_id]

        # Draw load below the node
        load_y = y - self.node_height / 2 - self.load_size - 0.15

        # Draw solid triangle pointing down (IEEE standard)
        triangle_points = [
            [x, load_y],
            [x - self.load_size / 2, load_y + self.load_size * 0.866],
            [x + self.load_size / 2, load_y + self.load_size * 0.866],
        ]
        triangle = mpatches.Polygon(triangle_points, edgecolor="black", facecolor="black", linewidth=2, zorder=2)
        self.ax.add_patch(triangle)

        # Draw connection line to node
        self.ax.plot([x, x], [y - self.node_height / 2, load_y + self.load_size * 0.866], "k-", linewidth=1.5, zorder=1)

        # Add label
        self.ax.text(x, load_y - 0.05, f"Load {load_id}", fontsize=7, ha="center", va="top")

    def _draw_generator(self, node_id: int, gen_id: int) -> None:
        """
        Draw a generator as a circle with 'G' (IEEE standard).

        Args:
            node_id: Node ID where the generator is connected
            gen_id: Generator identifier
        """
        if self.ax is None or node_id not in self.positions:
            return

        x, y = self.positions[node_id]

        # Draw generator below the node (offset horizontally to avoid overlap)
        gen_x = x + 0.4
        gen_y = y - self.node_height / 2 - self.gen_radius - 0.3

        # Draw circle
        circle = mpatches.Circle(
            (gen_x, gen_y), self.gen_radius, edgecolor="black", facecolor="white", linewidth=2, zorder=2
        )
        self.ax.add_patch(circle)

        # Draw 'G' inside with smaller font
        self.ax.text(gen_x, gen_y, "G", fontsize=6, ha="center", va="center", weight="bold")

        # Draw orthogonal connection line to node (horizontal then vertical)
        self.ax.plot([x, gen_x], [y - self.node_height / 2, y - self.node_height / 2], "k-", linewidth=1.5, zorder=1)
        self.ax.plot([gen_x, gen_x], [y - self.node_height / 2, gen_y + self.gen_radius], "k-", linewidth=1.5, zorder=1)

        # Add label
        self.ax.text(gen_x, gen_y - self.gen_radius - 0.05, f"Gen {gen_id}", fontsize=7, ha="center", va="top")

    def _draw_shunt(self, node_id: int, shunt_id: int) -> None:
        """
        Draw a shunt as capacitor symbol (two parallel lines).

        Args:
            node_id: Node ID where the shunt is connected
            shunt_id: Shunt identifier
        """
        if self.ax is None or node_id not in self.positions:
            return

        x, y = self.positions[node_id]

        # Draw shunt to the side of the node
        shunt_x = x - 0.5
        shunt_y = y - 0.5

        # Draw orthogonal connection line (horizontal then vertical)
        self.ax.plot([x - self.node_width / 2, shunt_x], [y, y], "k-", linewidth=1.5, zorder=1)
        self.ax.plot([shunt_x, shunt_x], [y, shunt_y], "k-", linewidth=1.5, zorder=1)

        # Draw capacitor symbol (two parallel lines)
        cap_height = 0.15
        self.ax.plot(
            [shunt_x - 0.05, shunt_x - 0.05],
            [shunt_y - cap_height / 2, shunt_y + cap_height / 2],
            "k-",
            linewidth=2,
            zorder=2,
        )
        self.ax.plot(
            [shunt_x + 0.05, shunt_x + 0.05],
            [shunt_y - cap_height / 2, shunt_y + cap_height / 2],
            "k-",
            linewidth=2,
            zorder=2,
        )

        # Draw ground symbol below capacitor
        ground_y = shunt_y - cap_height / 2 - 0.1
        # Ground line
        self.ax.plot([shunt_x, shunt_x], [shunt_y - cap_height / 2, ground_y], "k-", linewidth=1.5, zorder=1)
        # Ground symbol (three horizontal lines of decreasing length)
        self.ax.plot([shunt_x - 0.08, shunt_x + 0.08], [ground_y, ground_y], "k-", linewidth=2, zorder=2)
        self.ax.plot([shunt_x - 0.05, shunt_x + 0.05], [ground_y - 0.03, ground_y - 0.03], "k-", linewidth=2, zorder=2)
        self.ax.plot([shunt_x - 0.03, shunt_x + 0.03], [ground_y - 0.06, ground_y - 0.06], "k-", linewidth=2, zorder=2)

        # Add label
        self.ax.text(shunt_x, ground_y - 0.1, f"Shunt {shunt_id}", fontsize=7, ha="center", va="top")

    def _draw_components(self) -> None:
        """Draw all network components with their IEEE standard symbols."""
        if self.ax is None:
            return

        # Draw edges first (lines, transformers)
        for from_node, to_node, data in self.graph.edges(data=True):
            edge_type = data.get("type", "line")
            edge_id = data.get("id", 0)

            if edge_type == "line" or edge_type == "link":
                self._draw_line(from_node, to_node, edge_id)
            elif edge_type == "transformer":
                self._draw_transformer(from_node, to_node, edge_id)
            elif edge_type == "trafo3w_branch":
                # For three-winding transformers, use transformer symbol
                self._draw_transformer(from_node, to_node, edge_id)

        # Draw nodes
        for node_id, data in self.graph.nodes(data=True):
            if node_id in self.positions:
                node_type = data.get("type", "node")
                if node_type == "node":
                    x, y = self.positions[node_id]
                    self._draw_node(node_id, x, y)

        # Draw sources
        if "source" in self.network:
            for source in self.network["source"]:
                node_id = int(source["node"])
                self._draw_source(node_id)

        # Draw symmetric loads
        if "sym_load" in self.network:
            for i, load in enumerate(self.network["sym_load"]):
                node_id = int(load["node"])
                load_id = int(load["id"])
                self._draw_load(node_id, load_id)

        # Draw asymmetric loads
        if "asym_load" in self.network:
            for load in self.network["asym_load"]:
                node_id = int(load["node"])
                load_id = int(load["id"])
                self._draw_load(node_id, load_id)

        # Draw symmetric generators
        if "sym_gen" in self.network:
            for gen in self.network["sym_gen"]:
                node_id = int(gen["node"])
                gen_id = int(gen["id"])
                self._draw_generator(node_id, gen_id)

        # Draw asymmetric generators
        if "asym_gen" in self.network:
            for gen in self.network["asym_gen"]:
                node_id = int(gen["node"])
                gen_id = int(gen["id"])
                self._draw_generator(node_id, gen_id)

        # Draw shunts
        if "shunt" in self.network:
            for shunt in self.network["shunt"]:
                node_id = int(shunt["node"])
                shunt_id = int(shunt["id"])
                self._draw_shunt(node_id, shunt_id)

    def render(self) -> tuple[plt.Figure, plt.Axes]:
        """
        Render the complete network diagram.

        Returns:
            Tuple of (figure, axes) for further customization if needed
        """
        # Parse network and calculate layout
        self._parse_network_data()
        self._calculate_layout()

        # Create figure
        figsize = max(10, len(self.graph.nodes) * 0.8)
        self.fig, self.ax = plt.subplots(figsize=(figsize, figsize * 0.75))

        # Draw all components
        self._draw_components()

        # Configure axes
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.ax.set_title("Power Grid Network Diagram", fontsize=14, weight="bold", pad=20)

        # Auto-adjust limits with padding
        if self.positions:
            x_coords = [pos[0] for pos in self.positions.values()]
            y_coords = [pos[1] for pos in self.positions.values()]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            padding = max(1.0, (x_max - x_min) * 0.15)
            self.ax.set_xlim(x_min - padding, x_max + padding)
            self.ax.set_ylim(y_min - padding, y_max + padding)

        plt.tight_layout()

        return self.fig, self.ax

    def save_to_pdf(self, pdf_file_name: str) -> None:
        """
        Save the rendered diagram to a PDF file.

        Args:
            pdf_file_name: Path to output PDF file
        """
        if self.fig is None:
            raise RuntimeError("Diagram has not been rendered yet. Call render() first.")

        # Ensure .pdf extension
        output_path = Path(pdf_file_name)
        if output_path.suffix.lower() != ".pdf":
            output_path = output_path.with_suffix(".pdf")

        # Save to PDF
        with PdfPages(str(output_path)) as pdf:
            pdf.savefig(self.fig, bbox_inches="tight")

        plt.close(self.fig)


def print_network(network: SingleDataset, PDF_file_name: str) -> None:
    """
    Print a Power Grid Model network to a PDF file with an electrical diagram.

    This function creates an IEEE-standard electrical diagram visualization of the
    PGM network data and exports it to a PDF file. The diagram includes standard
    representations for components:
    - Buses/Nodes: Thick horizontal bars
    - Lines: Connecting lines between nodes
    - Transformers: Two circles (two-winding) or three circles (three-winding)
    - Sources: Circle with sine wave symbol
    - Loads: Triangle pointing down
    - Generators: Circle with 'G'
    - Shunts: Capacitor symbol

    Args:
        network: Power Grid Model network data (dictionary with component types as keys,
                 each containing arrays of component data)
        PDF_file_name: Output PDF file path (extension will be set to .pdf if not provided)

    Example:
        >>> from power_grid_model import PowerGridModel
        >>> from power_grid_model_io.converters import PgmJsonConverter
        >>> from power_grid_model_io.utils.print_network import print_network
        >>>
        >>> # Load network data
        >>> converter = PgmJsonConverter(source_file="network.json")
        >>> input_data, extra_info = converter.load_input_data()
        >>>
        >>> # Print network diagram to PDF
        >>> print_network(input_data, "network_diagram.pdf")

    Raises:
        ValueError: If network data is empty or invalid
        IOError: If unable to write to the specified PDF file
    """
    if not network:
        raise ValueError("Network data is empty")

    # Validate that we have at least nodes
    if "node" not in network or len(network["node"]) == 0:
        raise ValueError("Network must contain at least one node")

    # Create renderer and generate diagram
    renderer = NetworkDiagramRenderer(network)
    renderer.render()
    renderer.save_to_pdf(PDF_file_name)
