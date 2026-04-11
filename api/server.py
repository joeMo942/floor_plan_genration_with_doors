import os
import sys
import uuid
import asyncio
from typing import Dict, List, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from floorplan_generation.topology import generate_topology_from_form, visualize_agent_output
from floorplan_generation.inference import run_gsdiff_inference
from door_placement.pipeline import run_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tasks_db: Dict[str, dict] = {}

class RoomConfig(BaseModel):
    living_rooms: int = 1
    bedrooms: int = 2
    master_bedrooms: int = 1
    bathrooms: int = 2
    kitchens: int = 1
    storage: int = 0

class DoorRequest(BaseModel):
    plans: List[int]

@app.post("/api/topology")
def create_topology(config: RoomConfig):
    user_input = {
        "property_type": "apartment",
        "rooms": config.dict()
    }
    
    agent_nodes, agent_edges = generate_topology_from_form(user_input)
    os.makedirs("outputs/topology", exist_ok=True)
    vis_path = "outputs/topology/bubble_diagram.png"
    visualize_agent_output(agent_nodes, agent_edges, save_path=vis_path)
    
    return {
        "nodes": agent_nodes,
        "edges": agent_edges,
        "image_url": "/api/outputs/topology/bubble_diagram.png"
    }

async def run_inference_background(task_id: str, nodes: list, edges: list):
    try:
        tasks_db[task_id]["status"] = "processing"
        json_dir = "outputs/custom_jsons"
        output_dir = "outputs/custom_test"
        
        # Inference is blocking
        await asyncio.to_thread(
            run_gsdiff_inference,
            nodes, edges,
            output_dir=output_dir,
            json_dir=json_dir,
            num_samples=15
        )
        
        results = [
            {"id": i, "image_url": f"/api/outputs/custom_test/custom_pred_{i}.png"} 
            for i in range(15)
        ]
        
        tasks_db[task_id]["status"] = "completed"
        tasks_db[task_id]["results"] = results
    except Exception as e:
        tasks_db[task_id]["status"] = "failed"
        tasks_db[task_id]["error"] = str(e)

@app.post("/api/generate")
def generate_floorplans(payload: dict, background_tasks: BackgroundTasks):
    nodes = payload.get("nodes")
    edges = payload.get("edges")
    
    if not nodes or not edges:
        raise HTTPException(status_code=400, detail="Missing nodes or edges")
        
    task_id = str(uuid.uuid4())
    tasks_db[task_id] = {"status": "pending"}
    
    background_tasks.add_task(run_inference_background, task_id, nodes, edges)
    
    return {"task_id": task_id}

async def run_doors_background(task_id: str, plans: List[int]):
    try:
        tasks_db[task_id]["status"] = "processing"
        json_dir = "outputs/custom_jsons"
        door_output_dir = "outputs/final_door_placements"
        
        results = []
        for plan_num in plans:
            input_json = os.path.join(json_dir, f"custom_pred_{plan_num}.json")
            out_path = os.path.join(door_output_dir, f"plan_{plan_num}")
            
            if os.path.exists(input_json):
                await asyncio.to_thread(run_pipeline, input_json=input_json, output_dir=out_path)
                final_img_url = f"/api/outputs/final_door_placements/plan_{plan_num}/final_visualization.png"
                results.append({
                    "id": plan_num,
                    "image_url": final_img_url
                })
                
        tasks_db[task_id]["status"] = "completed"
        tasks_db[task_id]["results"] = results
    except Exception as e:
        tasks_db[task_id]["status"] = "failed"
        tasks_db[task_id]["error"] = str(e)

@app.post("/api/doors")
def generate_doors(request: DoorRequest, background_tasks: BackgroundTasks):
    if not request.plans:
        raise HTTPException(status_code=400, detail="No plans selected")
        
    task_id = str(uuid.uuid4())
    tasks_db[task_id] = {"status": "pending"}
    
    background_tasks.add_task(run_doors_background, task_id, request.plans)
    
    return {"task_id": task_id}

@app.get("/api/status/{task_id}")
def get_task_status(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks_db[task_id]

frontend_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web_ui", "dist")
outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

# Note: to prevent conflicts, I mounted output to /api/outputs
if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)

app.mount("/api/outputs", StaticFiles(directory=outputs_dir), name="outputs")

# Finally, mount static UI
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
else:
    @app.get("/")
    def no_frontend():
        return {"message": "Frontend not built yet. Please build the React UI into web_ui/dist."}
