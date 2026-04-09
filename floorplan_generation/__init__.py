"""
Floor Plan Generation Module.

Contains the logic to build the topology (bubble diagram) from user inputs,
and generate layout geometries by inferring through the GSDiff reverse 
diffusion process.
"""

from floorplan_generation.topology import generate_topology_from_form, visualize_agent_output
from floorplan_generation.inference import run_gsdiff_inference, generate_tensors_from_agent
