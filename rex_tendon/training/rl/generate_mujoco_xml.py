"""Generate MuJoCo XML models for tentacle robots."""

import math
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import Dict, List, Optional
import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="MuJoCo XML generation utilities")


def compute_joint_stiffness(
    i, n, base_diameter, base_thickness, tip_diameter, tip_thickness, E, damping_factor
):
    """Return (stiffness, damping) for segment i using geometric interpolation."""
    if n <= 1:
        return 0.3, 0.01

    alpha = i / float(n - 1)
    d_i = base_diameter * (tip_diameter / base_diameter) ** alpha
    t_i = base_thickness * (tip_thickness / base_thickness) ** alpha
    I_i = (math.pi * (d_i**4)) / 64.0
    K_i = (E * I_i) / t_i
    D_i = damping_factor * K_i
    return K_i, D_i


def generate_tentacle_xml():
    """Generate hardcoded tentacle MuJoCo XML model."""

    # Hardcoded configuration values
    rotation_offset_degrees = 60.0
    actuator_kp = 200.0
    actuator_force_range = "-200 0"
    tendon_rest_length = 0.34
    tendon_max_shortening = 0.22
    joint_range = "0 0.9"
    E = 2.0e7  # Young's modulus
    base_diameter = 3.4e-3  # m
    base_thickness = 0.93e-3  # m
    tip_diameter = 1.5e-3  # m
    tip_thickness = 0.50e-3  # m
    damping_factor = 0.07
    generate_contacts = False

    # Hardcoded joint positions (converted from mm to m)
    joint_z_positions = [
        14.3255e-3,
        34.275e-3,
        52.814e-3,
        70.0425e-3,
        86.0535e-3,
        100.9325e-3,
        114.76e-3,
        127.61e-3,
        139.55e-3,
        150.649e-3,
        160.962e-3,
        170.55e-3,
        179.453e-3,
        187.73e-3,
        195.42e-3,
        202.571e-3,
        209.2135e-3,
        215.3875e-3,
        221.1245e-3,
        226.4565e-3,
    ]

    # Hardcoded anchor positions (converted from mm to m)
    anchor_z_positions = [
        (5e-3, 12.02e-3),
        (20.421e-3, 32.131e-3),
        (39.939e-3, 50.821e-3),
        (58.077e-3, 68.19e-3),
        (74.932e-3, 84.331e-3),
        (90.597e-3, 99.331e-3),
        (105.154e-3, 113.271e-3),
        (118.682e-3, 126.225e-3),
        (131.254e-3, 138.264e-3),
        (142.938e-3, 149.452e-3),
        (153.795e-3, 159.849e-3),
        (163.885e-3, 169.511e-3),
        (173.262e-3, 178.49e-3),
        (181.976e-3, 186.835e-3),
        (190.074e-3, 194.59e-3),
        (197.60e-3, 201.79e-3),
        (204.594e-3, 208.493e-3),
        (211.093e-3, 214.717e-3),
        (217.133e-3, 220.501e-3),
        (222.746e-3, 225.876e-3),
        (227.963e-3, 230.871e-3),
    ]

    # Hardcoded excentric positions (converted from mm to m)
    excentric_y_positions_extremities = [13.74e-3, 2.35e-3]

    num_bodies = len(joint_z_positions) + 1
    ctrlrange_min = tendon_rest_length - tendon_max_shortening
    ctrlrange_max = tendon_rest_length
    actuator_ctrl_range = f"{ctrlrange_min:.3f} {ctrlrange_max:.3f}"

    # Create XML structure
    root = ET.Element("mujoco", attrib={"model": "tentacle"})
    ET.SubElement(root, "compiler", attrib={"angle": "radian", "autolimits": "true"})
    ET.SubElement(root, "option", attrib={"timestep": "0.005", "iterations": "50"})

    # Default settings
    default = ET.SubElement(root, "default")
    ET.SubElement(
        default,
        "joint",
        attrib={"stiffness": "0.0", "damping": "0.0", "range": joint_range},
    )
    default_geom_contype = "1" if generate_contacts else "0"
    default_geom_conaffinity = "1" if generate_contacts else "0"
    ET.SubElement(
        default,
        "geom",
        attrib={
            "type": "mesh",
            "xyaxes": "0 0 -1 -1 0 0",
            "contype": default_geom_contype,
            "conaffinity": default_geom_conaffinity,
            "margin": "0.001",
            "solref": "0.01 1",
            "solimp": "0.9 0.95 0.001",
        },
    )

    # Assets
    asset = ET.SubElement(root, "asset")
    for i in range(1, num_bodies + 1):
        ET.SubElement(
            asset,
            "mesh",
            attrib={
                "name": f"mesh{i}",
                "file": f"mujoco_rex_assets/Part{i}.stl",
                "scale": "0.001 0.001 0.001",
            },
        )

    # Textures and materials
    ET.SubElement(
        asset,
        "texture",
        attrib={
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.3 0.5 0.7",
            "rgb2": "0 0 0",
            "width": "512",
            "height": "3072",
        },
    )
    ET.SubElement(
        asset,
        "texture",
        attrib={
            "type": "2d",
            "name": "groundplane",
            "builtin": "checker",
            "mark": "edge",
            "rgb1": "0.2 0.3 0.4",
            "rgb2": "0.1 0.2 0.3",
            "markrgb": "0.8 0.8 0.8",
            "width": "300",
            "height": "300",
        },
    )
    ET.SubElement(
        asset,
        "material",
        attrib={
            "name": "groundplane",
            "texture": "groundplane",
            "texuniform": "true",
            "texrepeat": "5 5",
            "reflectance": "0.2",
        },
    )

    # Visual settings
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        attrib={
            "diffuse": "0.6 0.6 0.6",
            "ambient": "0.3 0.3 0.3",
            "specular": "0 0 0",
        },
    )
    ET.SubElement(visual, "rgba", attrib={"haze": "0.15 0.25 0.35 1"})
    ET.SubElement(visual, "global", attrib={"azimuth": "150", "elevation": "-20"})

    # World body
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        attrib={"pos": "0 0 3", "dir": "0 0 -1", "directional": "false"},
    )

    floor_conaffinity = "15" if generate_contacts else "0"
    ET.SubElement(
        worldbody,
        "geom",
        attrib={
            "name": "floor",
            "pos": "0 0 -0.2",
            "size": "0 0 .125",
            "type": "plane",
            "material": "groundplane",
            "conaffinity": floor_conaffinity,
            "condim": "3",
            "xyaxes": "1 0 0 0 1 0",
        },
    )

    # Cameras
    ET.SubElement(
        worldbody,
        "camera",
        attrib={
            "name": "fixed_overview",
            "mode": "fixed",
            "pos": "0 -0.5 0.7",
            "quat": "0.924 0.383 0 0",
            "fovy": "60",
        },
    )
    ET.SubElement(
        worldbody,
        "camera",
        attrib={
            "name": "tracking_cam",
            "mode": "fixed",
            "pos": "0.3 -0.3 0.4",
            "quat": "0.854 0.354 0.146 0.354",
            "fovy": "75",
        },
    )

    # Target site
    ET.SubElement(
        worldbody,
        "site",
        attrib={
            "name": "target",
            "type": "sphere",
            "size": "0.01 0.01 0.01",
            "rgba": "1 0 0 0.8",
            "pos": "0 0 0",
        },
    )

    # Create body chain
    body_refs = []
    body_geom_contype = "1" if generate_contacts else "0"
    body_geom_conaffinity = "1" if generate_contacts else "0"

    # First body
    body1 = ET.SubElement(worldbody, "body", attrib={"name": "body1", "pos": "0 0 0"})
    ET.SubElement(
        body1,
        "geom",
        attrib={
            "name": "geom1",
            "type": "mesh",
            "mesh": "mesh1",
            "contype": body_geom_contype,
            "conaffinity": body_geom_conaffinity,
        },
    )
    body_refs.append(body1)
    current_body = body1

    # Subsequent bodies with joints
    for i, z in enumerate(joint_z_positions):
        K_i, D_i = compute_joint_stiffness(
            i,
            len(joint_z_positions),
            base_diameter,
            base_thickness,
            tip_diameter,
            tip_thickness,
            E,
            damping_factor,
        )

        new_body = ET.SubElement(
            current_body, "body", attrib={"name": f"body{i+2}", "pos": "0 0 0"}
        )
        ET.SubElement(
            new_body,
            "geom",
            attrib={
                "name": f"geom{i+2}",
                "type": "mesh",
                "mesh": f"mesh{i+2}",
                "contype": body_geom_contype,
                "conaffinity": body_geom_conaffinity,
            },
        )
        ET.SubElement(
            new_body,
            "joint",
            attrib={
                "type": "ball",
                "name": f"joint{i+1}",
                "pos": f"0 0 {z}",
                "stiffness": f"{K_i:.5g}",
                "damping": f"{D_i:.5g}",
            },
        )

        body_refs.append(new_body)
        current_body = new_body

    # Tendon geometry
    offset_rad = math.radians(rotation_offset_degrees)
    angles = [
        offset_rad,
        offset_rad + 2.0 * math.pi / 3.0,
        offset_rad + 4.0 * math.pi / 3.0,
    ]

    n_pairs = len(anchor_z_positions)
    z_global_start = anchor_z_positions[0][0]
    z_global_end = anchor_z_positions[-1][1]
    dz_global = z_global_end - z_global_start
    r_start, r_end = excentric_y_positions_extremities
    tendon_site_names = [[], [], []]

    def global_line_xyz(k, t):
        """Global 3D line for tendon k, param t in [0..1]. Linear in z, radius, angle fixed."""
        z_val = z_global_start + t * dz_global
        x_val_start = r_start * math.cos(angles[k])
        x_val_end = r_end * math.cos(angles[k])
        x_val = x_val_start + t * (x_val_end - x_val_start)
        y_val_start = r_start * math.sin(angles[k])
        y_val_end = r_end * math.sin(angles[k])
        y_val = y_val_start + t * (y_val_end - y_val_start)
        return (x_val, y_val, z_val)

    # Create tendon sites
    for i in range(n_pairs):
        body_i = body_refs[i]
        z_in_local, z_out_local = anchor_z_positions[i]
        t_in = (z_in_local - z_global_start) / dz_global
        t_out = (z_out_local - z_global_start) / dz_global

        for k in range(3):
            x_in, y_in, z_in = global_line_xyz(k, t_in)
            x_out, y_out, z_out = global_line_xyz(k, t_out)

            site_in_name = f"site_in_{i}_{k}"
            site_out_name = f"site_out_{i}_{k}"

            ET.SubElement(
                body_i,
                "site",
                attrib={
                    "name": site_in_name,
                    "pos": f"{x_in:.5f} {y_in:.5f} {z_in:.5f}",
                    "size": "0.001",
                    "rgba": "1 1 0 1",
                },
            )
            tendon_site_names[k].append(site_in_name)

            ET.SubElement(
                body_i,
                "site",
                attrib={
                    "name": site_out_name,
                    "pos": f"{x_out:.5f} {y_out:.5f} {z_out:.5f}",
                    "size": "0.001",
                    "rgba": "1 1 0 1",
                },
            )
            tendon_site_names[k].append(site_out_name)

    # Add tip center site to last body
    if body_refs:
        last_body = body_refs[-1]
        ET.SubElement(
            last_body,
            "site",
            attrib={
                "name": "tip_center",
                "pos": f"0 0 {anchor_z_positions[-1][1]}",
                "size": "0.002",
                "rgba": "0 1 0 1",
            },
        )

    # Create tendons
    tendon_elem = ET.SubElement(root, "tendon")
    for k in range(3):
        spatial_tendon = ET.SubElement(
            tendon_elem,
            "spatial",
            attrib={"name": f"tendon_{k+1}", "width": "0.001", "rgba": "1 0 0 1"},
        )
        for site_name in tendon_site_names[k]:
            ET.SubElement(spatial_tendon, "site", attrib={"site": site_name})

    # Create actuators
    actuator_elem = ET.SubElement(root, "actuator")
    for k in range(3):
        ET.SubElement(
            actuator_elem,
            "position",
            attrib={
                "name": f"actuator_{k+1}",
                "tendon": f"tendon_{k+1}",
                "kp": str(actuator_kp),
                "forcerange": actuator_force_range,
                "ctrlrange": actuator_ctrl_range,
            },
        )

    # Create sensors
    sensor_elem = ET.SubElement(root, "sensor")
    for k in range(3):
        ET.SubElement(
            sensor_elem,
            "tendonpos",
            attrib={"name": f"tendon{k+1}_pos", "tendon": f"tendon_{k+1}"},
        )

    # Contact exclusions
    if generate_contacts:
        contact_elem = ET.SubElement(root, "contact")
        for i in range(1, num_bodies):
            ET.SubElement(
                contact_elem,
                "exclude",
                attrib={"body1": f"body{i}", "body2": f"body{i+1}"},
            )

    # Convert to pretty XML
    rough_string = ET.tostring(root, encoding="unicode")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


@app.command()
def generate_xml(
    output_path: str = typer.Option(
        "rex_assets/rex_simulation/tentacle.xml", "--output-path", help="Output XML file path"
    ),
) -> None:
    """Generate MuJoCo XML model."""
    xml_content = generate_tentacle_xml()
    output_path = Path(output_path)

    with open(output_path, "w") as f:
        f.write(xml_content)

    console.print(f"MuJoCo XML file created: {output_path}")


if __name__ == "__main__":
    app()

