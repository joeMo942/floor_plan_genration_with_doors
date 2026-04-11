"""
Topology and Bubble Diagram Generator.

Provides the logic to read unstructured user requirements (rooms, counts)
and output a connected Graph (Nodes & Edges) representing a valid 
architectural bubble diagram.
"""

import random
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_topology_from_form(user_input: dict) -> tuple[list[int], list[list[int]]]:
    """
    Generates topology with a Living Room Chain (Public to Private transition).
    Room IDs: 0: Living, 1: Bedroom, 2: Storage, 3: Kitchen, 4: Bathroom
    """
    rng = random.SystemRandom() 
    rooms_req = user_input.get("rooms", {})
    nodes = []
    edges = []
    
    def add_node(room_type):
        nodes.append(room_type)
        return len(nodes) - 1

    # ==========================================
    # 1. THE LIVING ROOM CHAIN
    # ==========================================
    num_living = max(1, rooms_req.get("living_rooms", 1))
    living_indices = []
    
    for _ in range(num_living):
        living_indices.append(add_node(0))
        
    if num_living > 1:
        for i in range(num_living - 1):
            edges.append([living_indices[i], living_indices[i+1]])

    public_hub = living_indices[0]        # Connects to Kitchens/Public
    private_hub = living_indices[-1]      # Connects to Beds/Private

    # ==========================================
    # 2. PUBLIC WING (Connected to Public Hub)
    # ==========================================
    kitchen_indices = []
    for _ in range(rooms_req.get("kitchens", 0)):
        k_idx = add_node(3)
        edges.append([public_hub, k_idx])
        kitchen_indices.append(k_idx)

    num_baths = rooms_req.get("bathrooms", 0)
    private_bath_count = 0
    if num_baths > 0:
        # Bath 1 is Public (Wet Wall)
        public_bath_idx = add_node(4)
        edges.append([public_hub, public_bath_idx])
        if kitchen_indices:
            edges.append([kitchen_indices[0], public_bath_idx])
        private_bath_count = num_baths - 1

    # ==========================================
    # 3. PRIVATE WING (Connected to Private Hub)
    # ==========================================
    num_beds = rooms_req.get("bedrooms", 0)
    bed_indices = []
    for _ in range(num_beds):
        b_idx = add_node(1)
        edges.append([private_hub, b_idx]) 
        bed_indices.append(b_idx)

    if num_beds > 1:
        for i in range(1, num_beds):
            target_bed = rng.choice(bed_indices[:i])
            edges.append([bed_indices[i], target_bed])

    # ==========================================
    # 4. MASTER SUITES & HALL BATHROOMS
    # ==========================================
    num_masters_requested = rooms_req.get("master_bedrooms", 0)
    
    if private_bath_count > 0:
        actual_masters = min(private_bath_count, num_masters_requested, num_beds)
        hall_baths = private_bath_count - actual_masters
        
        shuffled_beds = bed_indices.copy()
        rng.shuffle(shuffled_beds)
        
        master_beds = shuffled_beds[:actual_masters]
        secondary_beds = shuffled_beds[actual_masters:]
        
        for m_bed in master_beds:
            en_suite_idx = add_node(4)
            edges.append([m_bed, en_suite_idx])
            
        for _ in range(hall_baths):
            hall_bath_idx = add_node(4)
            edges.append([private_hub, hall_bath_idx])
            if secondary_beds:
                edges.append([rng.choice(secondary_beds), hall_bath_idx])
            elif master_beds:
                edges.append([rng.choice(master_beds), hall_bath_idx])

    # ==========================================
    # 5. STORAGE
    # ==========================================
    for _ in range(rooms_req.get("storage", 0)):
        stor_idx = add_node(2)
        edges.append([public_hub, stor_idx])

    return nodes, edges

def visualize_agent_output(nodes: list[int], edges: list[list[int]], save_path: str = None):
    """
    Visualizes the generated topology using a force-directed layout and
    optionally saves the plot to a file if save_path is provided.
    """
    room_mapping = {0: "Living", 1: "Bedroom", 2: "Storage", 3: "Kitchen", 4: "Bathroom", 5: "Balcony"}
    color_map = {0: '#F4F1DE', 1: '#EAB69F', 2: '#6B705C', 3: '#E07A5F', 4: '#5F797B', 5: '#F2CC8F'}

    G = nx.Graph()
    node_colors = []
    labels = {}

    for i, room_type in enumerate(nodes):
        G.add_node(i)
        node_colors.append(color_map.get(room_type, '#CCCCCC'))
        labels[i] = f"[{i}]\n{room_mapping.get(room_type, 'Unknown')}"

    for u, v in edges:
        G.add_edge(u, v)

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, k=0.9, iterations=50) 
    nx.draw(G, pos, with_labels=True, labels=labels, node_color=node_colors, 
            node_size=3000, font_size=9, font_weight="bold", edge_color="#555555", width=2.5)
    plt.title("AI Generated Topology (Bubble Diagram)", fontsize=14, fontweight="bold")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Topology visualization saved to {save_path}")
    else:
        plt.show()
    plt.close()
