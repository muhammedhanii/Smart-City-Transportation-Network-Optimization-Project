"""
MST Infrastructure Module

This module uses a modified version of Kruskal's algorithm to build a
cost-efficient road network for the Smart City Transportation Network project.

The code is intentionally simple and readable for a university project.
"""

from dataclasses import dataclass

CRITICAL_FACILITY_TYPES = {"Medical", "Government", "Transit Hub", "Airport"}

# Default critical facilities:
# F1 = Cairo International Airport
# F2 = Ramses Railway Station
# F9 = Qasr El Aini Hospital
# F10 = Maadi Military Hospital
# 13 = New Administrative Capital
DEFAULT_CRITICAL_FACILITIES = ["F1", "F2", "F9", "F10", "13"]

@dataclass
class Edge:
    from_id: str
    to_id: str
    distance: float
    capacity: int
    condition: int
    construction_cost: float
    road_type: str  # "existing" or "new"
    weight: float = 0

class UnionFind:
    """Simple Union-Find structure used to detect cycles in Kruskal's algorithm."""

    def __init__(self, nodes):
        self.parent = {}
        self.rank = {}

        for node in nodes:
            node_id = get_node_id(node)
            self.parent[node_id] = node_id
            self.rank[node_id] = 0

    def find(self, node_id):
        if self.parent[node_id] != node_id:
            self.parent[node_id] = self.find(self.parent[node_id])
        return self.parent[node_id]

    def union(self, node_a, node_b):
        root_a = self.find(node_a)
        root_b = self.find(node_b)

        if root_a == root_b:
            return False

        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1

        return True

def get_node_id(node):
    """Return node id whether the node is a dictionary or a simple object."""
    if isinstance(node, dict):
        return str(node["id"])
    return str(node.id)

def get_value(item, key):
    """Return a value from a dictionary or object."""
    if isinstance(item, dict):
        return item[key]
    return getattr(item, key)

def get_edge_weight(edge):
    """Return edge weight safely, even if a dictionary edge has no weight yet."""
    if isinstance(edge, dict):
        return edge.get("weight", 0)
    return getattr(edge, "weight", 0)

def set_edge_weight(edge, weight):
    """Set edge weight whether the edge is a dictionary or an Edge object."""
    if isinstance(edge, dict):
        edge["weight"] = weight
    else:
        edge.weight = weight

def get_node_map(nodes):
    """Convert node list into a dictionary: node_id -> node."""
    node_map = {}
    for node in nodes:
        node_map[get_node_id(node)] = node
    return node_map

def edge_key(edge):
    """Create a unique key for an undirected edge."""
    from_id = str(get_value(edge, "from_id"))
    to_id = str(get_value(edge, "to_id"))
    return tuple(sorted([from_id, to_id]))

def calculate_mst_weight(edge, nodes):
    """
    Calculate the modified MST weight.

    Formula:
    weight = cost_score + distance_score - population_bonus - facility_bonus
    """
    node_map = get_node_map(nodes)
    from_id = str(get_value(edge, "from_id"))
    to_id = str(get_value(edge, "to_id"))

    from_node = node_map[from_id]
    to_node = node_map[to_id]

    distance = get_value(edge, "distance")
    road_type = get_value(edge, "road_type")
    condition = get_value(edge, "condition")
    construction_cost = get_value(edge, "construction_cost")

    distance_score = distance * 10

    if road_type == "existing":
        cost_score = (11 - condition) * 20
    else:
        cost_score = construction_cost

    population_bonus = 0
    if get_value(from_node, "population") >= 300000:
        population_bonus += 50
    if get_value(to_node, "population") >= 300000:
        population_bonus += 50

    facility_bonus = 0
    if get_value(from_node, "type") in CRITICAL_FACILITY_TYPES:
        facility_bonus += 80
    if get_value(to_node, "type") in CRITICAL_FACILITY_TYPES:
        facility_bonus += 80

    weight = cost_score + distance_score - population_bonus - facility_bonus
    return max(weight, 1)

def kruskal_mst(nodes, edges):
    """
    Build the MST using Kruskal's algorithm.

    Steps:
    1. Calculate weight for every edge.
    2. Sort edges by weight.
    3. Use Union-Find to avoid cycles.
    4. Stop when all nodes are connected.
    """
    union_find = UnionFind(nodes)
    mst_edges = []

    for edge in edges:
        weight = calculate_mst_weight(edge, nodes)
        if isinstance(edge, dict):
            edge["weight"] = weight
        else:
            edge.weight = weight

    sorted_edges = sorted(edges, key=lambda edge: get_value(edge, "weight"))

    for edge in sorted_edges:
        from_id = str(get_value(edge, "from_id"))
        to_id = str(get_value(edge, "to_id"))

        if union_find.union(from_id, to_id):
            mst_edges.append(edge)

        if len(mst_edges) == len(nodes) - 1:
            break

    return mst_edges

def is_network_connected(nodes, mst_edges):
    """Return True if the selected edges connect all nodes, otherwise False."""
    if not nodes:
        return True

    union_find = UnionFind(nodes)

    for edge in mst_edges:
        from_id = str(get_value(edge, "from_id"))
        to_id = str(get_value(edge, "to_id"))

        if from_id in union_find.parent and to_id in union_find.parent:
            union_find.union(from_id, to_id)

    first_node_id = get_node_id(nodes[0])
    first_root = union_find.find(first_node_id)

    for node in nodes:
        if union_find.find(get_node_id(node)) != first_root:
            return False

    return True

def count_facility_connections(facility_id, mst_edges):
    """Count how many direct MST connections a facility has."""
    count = 0

    for edge in mst_edges:
        from_id = str(get_value(edge, "from_id"))
        to_id = str(get_value(edge, "to_id"))

        if facility_id == from_id or facility_id == to_id:
            count += 1

    return count

def ensure_critical_connectivity(mst_edges, all_edges, critical_facilities, nodes=None):
    """
    Make sure every critical facility has at least two direct connections if possible.

    If a facility has less than 2 direct connections, this function adds the
    cheapest available edge connected to that facility that is not already in
    the MST.
    """
    selected_edge_keys = set()
    unique_mst_edges = []

    for edge in mst_edges:
        key = edge_key(edge)
        if key not in selected_edge_keys:
            unique_mst_edges.append(edge)
            selected_edge_keys.add(key)

    mst_edges = unique_mst_edges

    for facility_id in critical_facilities:
        facility_id = str(facility_id)

        while count_facility_connections(facility_id, mst_edges) < 2:
            possible_edges = []

            for edge in all_edges:
                from_id = str(get_value(edge, "from_id"))
                to_id = str(get_value(edge, "to_id"))

                connects_to_facility = facility_id == from_id or facility_id == to_id
                not_selected = edge_key(edge) not in selected_edge_keys

                if connects_to_facility and not_selected:
                    if get_edge_weight(edge) == 0 and nodes is not None:
                        set_edge_weight(edge, calculate_mst_weight(edge, nodes))
                    possible_edges.append(edge)

            if not possible_edges:
                break

            cheapest_edge = min(
                possible_edges,
                key=lambda edge: get_edge_weight(edge)
                if get_edge_weight(edge) != 0
                else get_value(edge, "construction_cost"),
            )

            mst_edges.append(cheapest_edge)
            selected_edge_keys.add(edge_key(cheapest_edge))

    return mst_edges

def calculate_cost_analysis(mst_edges):
    """Calculate basic cost and maintenance information for the selected roads."""
    total_distance = 0
    total_construction_cost = 0
    number_of_existing_roads = 0
    number_of_new_roads = 0
    maintenance_score = 0

    for edge in mst_edges:
        distance = get_value(edge, "distance")
        road_type = get_value(edge, "road_type")
        condition = get_value(edge, "condition")

        total_distance += distance

        if road_type == "existing":
            number_of_existing_roads += 1
            maintenance_score += (11 - condition) * distance
        else:
            number_of_new_roads += 1
            total_construction_cost += get_value(edge, "construction_cost")

    return {
        "total_distance_km": total_distance,
        "total_construction_cost_million_egp": total_construction_cost,
        "number_of_existing_roads": number_of_existing_roads,
        "number_of_new_roads": number_of_new_roads,
        "maintenance_score": maintenance_score,
    }

def print_mst_result(mst_edges):
    """Print selected MST roads and a simple cost analysis."""
    print("Selected roads in MST:")
    print("-" * 60)

    for edge in mst_edges:
        print(
            f"{get_value(edge, 'from_id')} -> {get_value(edge, 'to_id')} | "
            f"distance: {get_value(edge, 'distance')} km | "
            f"type: {get_value(edge, 'road_type')} | "
            f"weight: {get_value(edge, 'weight')}"
        )

    analysis = calculate_cost_analysis(mst_edges)

    print("\nCost analysis:")
    print("-" * 60)
    print(f"Total distance: {analysis['total_distance_km']} km")
    print(
        "Total construction cost: "
        f"{analysis['total_construction_cost_million_egp']} Million EGP"
    )
    print(f"Existing roads: {analysis['number_of_existing_roads']}")
    print(f"New roads: {analysis['number_of_new_roads']}")
    print(f"Maintenance score: {analysis['maintenance_score']}")

def get_mst_report_note():
    """Return a short explanation of the constrained MST approach."""
    return (
        "We first generated an MST using a modified Kruskal algorithm. Then, "
        "we added extra edges when needed to satisfy critical facility "
        "connectivity constraints. Therefore, the final network is a "
        "constrained MST-based optimized network, not always a pure MST."
    )

def get_mst_complexity_analysis():
    """Return the time and space complexity of the MST algorithm."""
    return (
        "The time complexity of Kruskal's algorithm is O(E log E), mainly due "
        "to sorting the edges. The Union-Find operations are almost constant "
        "time, so the total complexity remains O(E log E). The space "
        "complexity is O(V + E)."
    )

def load_real_project_data():
    """
    Placeholder for connecting this module with the real Cairo project data loader.
    The main app should pass real nodes and edges into kruskal_mst().
    """
    pass

def visualize_mst(nodes, mst_edges):
    """
    Draw the MST using networkx and matplotlib.

    Install required libraries if needed:
    pip install networkx matplotlib
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("Visualization requires networkx and matplotlib.")
        print("Install them using: pip install networkx matplotlib")
        return

    graph = nx.Graph()
    positions = {}
    labels = {}

    for node in nodes:
        node_id = get_node_id(node)
        graph.add_node(node_id)
        positions[node_id] = (get_value(node, "x"), get_value(node, "y"))
        labels[node_id] = get_value(node, "name")

    for edge in mst_edges:
        graph.add_edge(str(get_value(edge, "from_id")), str(get_value(edge, "to_id")))

    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(graph, positions, node_size=700, node_color="lightblue")
    nx.draw_networkx_edges(graph, positions, width=2, edge_color="darkgreen")
    nx.draw_networkx_labels(graph, positions, labels, font_size=9)

    plt.title("Optimized Road Network using Modified Kruskal MST")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    sample_nodes = [
        {
            "id": "F1",
            "name": "Cairo Airport",
            "population": 100000,
            "type": "Airport",
            "x": 8,
            "y": 9,
        },
        {
            "id": "F2",
            "name": "Ramses Station",
            "population": 250000,
            "type": "Transit Hub",
            "x": 4,
            "y": 6,
        },
        {
            "id": "F9",
            "name": "Major Hospital",
            "population": 150000,
            "type": "Medical",
            "x": 2,
            "y": 4,
        },
        {
            "id": "F10",
            "name": "Maadi Military Hospital",
            "population": 150000,
            "type": "Medical",
            "x": 6,
            "y": 3,
        },
        {
            "id": "13",
            "name": "New Administrative Capital",
            "population": 50000,
            "type": "Government",
            "x": 7,
            "y": 6,
        },
        {
            "id": "20",
            "name": "Maadi",
            "population": 320000,
            "type": "Residential",
            "x": 3,
            "y": 1,
        },
    ]

    sample_edges = [
        Edge("F1", "13", 8.0, 4000, 8, 600, "existing"),
        Edge("F1", "F2", 12.0, 3000, 7, 900, "existing"),
        Edge("F1", "F10", 9.0, 3500, 6, 800, "new"),
        Edge("F2", "F9", 4.5, 5000, 9, 300, "existing"),
        Edge("F2", "13", 5.0, 4500, 7, 400, "existing"),
        Edge("F9", "F10", 7.0, 2500, 5, 700, "new"),
        Edge("F9", "20", 4.0, 3000, 8, 350, "existing"),
        Edge("F10", "13", 3.5, 4000, 6, 500, "existing"),
        Edge("F10", "20", 6.0, 3500, 7, 450, "new"),
        Edge("13", "20", 7.5, 3000, 8, 550, "existing"),
    ]

    mst = kruskal_mst(sample_nodes, sample_edges)
    mst = ensure_critical_connectivity(
        mst, sample_edges, DEFAULT_CRITICAL_FACILITIES, sample_nodes
    )

    print_mst_result(mst)
    print(f"\nNetwork connected: {is_network_connected(sample_nodes, mst)}")
    print(f"\nReport note:\n{get_mst_report_note()}")
    print(f"\nComplexity analysis:\n{get_mst_complexity_analysis()}")
    visualize_mst(sample_nodes, mst)


