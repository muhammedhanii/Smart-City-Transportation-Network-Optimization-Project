"""Unit tests for the DataLoader class."""

import json
import pytest
from pathlib import Path
from data.data_loader import DataLoader, DataLoadError
from core.graph import Graph
from utils.enums import NodeType, TimeOfDay


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_data"


@pytest.fixture()
def loader() -> DataLoader:
    return DataLoader(data_dir=_SAMPLE_DIR)


@pytest.fixture()
def graph(loader: DataLoader) -> Graph:
    return loader.load()


# ─────────────────────────────────────────────────────────────────────────────
# Load tests
# ─────────────────────────────────────────────────────────────────────────────


def test_load_returns_graph(loader: DataLoader) -> None:
    result = loader.load()
    assert isinstance(result, Graph)


def test_graph_has_25_nodes(graph: Graph) -> None:
    """15 neighborhoods + 10 facilities = 25 nodes."""
    assert graph.node_count == 25


def test_graph_has_expected_edge_count(graph: Graph) -> None:
    """37 current roads + 2 proposed roads = 39 edges total."""
    assert graph.edge_count == 39


def test_graph_has_2_hospitals(graph: Graph) -> None:
    assert len(graph.get_hospitals()) == 2


def test_graph_has_2_fire_stations(graph: Graph) -> None:
    fire_stations = graph.get_facilities_by_type(NodeType.FIRE_STATION)
    assert len(fire_stations) == 2


def test_graph_has_2_police_stations(graph: Graph) -> None:
    police = graph.get_facilities_by_type(NodeType.POLICE)
    assert len(police) == 2


def test_graph_has_2_schools(graph: Graph) -> None:
    schools = graph.get_facilities_by_type(NodeType.SCHOOL)
    assert len(schools) == 2


def test_graph_has_2_transit_hubs(graph: Graph) -> None:
    hubs = graph.get_facilities_by_type(NodeType.TRANSIT_HUB)
    assert len(hubs) == 2


def test_graph_has_15_neighborhoods(graph: Graph) -> None:
    neighborhoods = graph.get_facilities_by_type(NodeType.NEIGHBORHOOD)
    assert len(neighborhoods) == 15


def test_dynamic_edge_weight_changes_by_time(graph: Graph) -> None:
    morning = graph.get_dynamic_edge_weight("N01", "N02", TimeOfDay.MORNING)
    night = graph.get_dynamic_edge_weight("N01", "N02", TimeOfDay.NIGHT)
    assert morning > night


def test_validate_passes_on_loaded_graph(graph: Graph) -> None:
    """Loaded sample data must pass structural validation."""
    graph.validate()


def test_public_transport_loaded(loader: DataLoader) -> None:
    loader.load()
    transport = loader.get_public_transport()
    assert "metro_lines" in transport
    assert "bus_routes" in transport


def test_public_transport_has_metro_lines(loader: DataLoader) -> None:
    loader.load()
    transport = loader.get_public_transport()
    assert len(transport["metro_lines"]) >= 1


def test_public_transport_has_bus_routes(loader: DataLoader) -> None:
    loader.load()
    transport = loader.get_public_transport()
    assert len(transport["bus_routes"]) >= 1


def test_exclude_proposed_roads() -> None:
    loader = DataLoader(data_dir=_SAMPLE_DIR, include_proposed=False)
    graph = loader.load()
    # Without proposed roads there should be 2 fewer edges.
    assert graph.edge_count == 37


def test_missing_data_dir_raises(tmp_path: Path) -> None:
    loader = DataLoader(data_dir=tmp_path / "nonexistent")
    with pytest.raises(DataLoadError):
        loader.load()


def test_malformed_json_raises(tmp_path: Path) -> None:
    """A corrupt JSON file must raise DataLoadError, not crash silently."""
    bad_nodes = tmp_path / "nodes.json"
    bad_nodes.write_text("{ this is not valid json }", encoding="utf-8")
    loader = DataLoader(data_dir=tmp_path)
    with pytest.raises(DataLoadError):
        loader.load()


def test_repr_contains_data_dir(loader: DataLoader) -> None:
    assert "sample_data" in repr(loader)
