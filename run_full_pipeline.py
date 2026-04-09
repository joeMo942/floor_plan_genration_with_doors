import os
import sys

# Add the local directory to python path so it can find the packages
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from floorplan_generation.topology import generate_topology_from_form, visualize_agent_output
from floorplan_generation.inference import run_gsdiff_inference
from door_placement.pipeline import run_pipeline

def main():
    print("="*60)
    print("     AI FLOOR PLAN GENERATION & DOOR PLACEMENT PIPELINE")
    print("="*60)
    
    # ---------------------------------------------------------
    # STEP 1: Take Room Counts
    # ---------------------------------------------------------
    print("\n[STEP 1] Please enter the number of rooms for your layout:")
    def get_input(prompt, default_val):
        val = input(f"{prompt} [default: {default_val}]: ").strip()
        return int(val) if val else default_val

    living_rooms    = get_input("  - Living Rooms   ", 1)
    bedrooms        = get_input("  - Bedrooms       ", 2)
    master_bedrooms = get_input("  - Master Bedrooms", 1)
    bathrooms       = get_input("  - Bathrooms      ", 2)
    kitchens        = get_input("  - Kitchens       ", 1)
    storage         = get_input("  - Storage Rooms  ", 0)
    
    user_input = {
        "property_type": "apartment",
        "rooms": {
            "living_rooms": living_rooms,
            "bedrooms": bedrooms,
            "master_bedrooms": master_bedrooms,
            "bathrooms": bathrooms,
            "kitchens": kitchens,
            "storage": storage
        }
    }
    
    # ---------------------------------------------------------
    # STEP 2: Generate Bubble Diagram (Topology)
    # ---------------------------------------------------------
    print("\n[STEP 2] Generating procedural topology (Bubble Diagram)...")
    agent_nodes, agent_edges = generate_topology_from_form(user_input)
    print(f"  -> Generated Nodes: {agent_nodes}")
    print(f"  -> Generated Edges: {agent_edges}")
    
    os.makedirs("outputs/custom_test", exist_ok=True)
    vis_path = "outputs/custom_test/bubble_diagram.png"
    visualize_agent_output(agent_nodes, agent_edges, save_path=vis_path)
    
    # ---------------------------------------------------------
    # STEP 3: Generate Floor Plans (GSDiff Inference)
    # ---------------------------------------------------------
    num_samples = 15
    json_dir = "outputs/custom_jsons"
    print(f"\n[STEP 3] Running GSDiff Generation ({num_samples} samples)...")
    print("This will process through the diffusion model. Please wait...")
    run_gsdiff_inference(
        agent_nodes, 
        agent_edges, 
        output_dir="outputs/custom_test", 
        json_dir=json_dir, 
        num_samples=num_samples
    )
    
    # ---------------------------------------------------------
    # STEP 4: Select Plans & Apply Door Placement
    # ---------------------------------------------------------
    print(f"\n[STEP 4] Generation Complete. Plans saved to '{json_dir}/'")
    plans_str = input(
        "\nEnter the plan numbers you want to apply door placement to,\n"
        "separated by comma (e.g. 0,3,14) [default: 0]: "
    )
    
    if not plans_str.strip():
        plans_str = "0"
        
    selected_plans = [p.strip() for p in plans_str.split(',')]
    
    door_output_dir = "outputs/final_door_placements"
    for plan_num in selected_plans:
        input_json = os.path.join(json_dir, f"custom_pred_{plan_num}.json")
        out_path = os.path.join(door_output_dir, f"plan_{plan_num}")
        
        if os.path.exists(input_json):
            print(f"\n>>> Running Door Placement on {input_json} >>>")
            run_pipeline(input_json=input_json, output_dir=out_path)
        else:
            print(f"  [!] Warning: {input_json} does not exist. Skipping.")

    print("\n" + "="*60)
    print("ALL DONE! Check the 'outputs/final_door_placements' folder.")
    print("="*60)


if __name__ == "__main__":
    main()
