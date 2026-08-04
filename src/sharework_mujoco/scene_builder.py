"""
Costruisce lo spec MuJoCo completo della cella sharework (fixed_parts +
UR10e + Robotiq 2F-85 + camere rs1/rs2/wrist), sia in memoria (per uso
diretto in Python, es. dentro sim.py) sia salvato su disco come MJCF
standalone (per uso con `simulate`, il viewer, o altri tool esterni).

Tutta la logica di attach/composizione pose e' la stessa gia' validata in
build_scene.py; qui e' solo riorganizzata in funzioni riusabili.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mujoco

ASSETS_DIR = Path(__file__).parent / "assets"
FIXED_PARTS_XML = ASSETS_DIR / "fixed_parts.xml"
UR10E_XML = ASSETS_DIR / "menagerie_src" / "ur10e.xml"
GRIPPER_XML = ASSETS_DIR / "menagerie_src" / "2f85.xml"
UR10E_MESHDIR = ASSETS_DIR / "meshes" / "ur10e"
GRIPPER_MESHDIR = ASSETS_DIR / "meshes" / "2f85"


def _compose(pos1, quat1, pos2, quat2):
    """pose risultante = pose1 * pose2 (pose2 espressa nel frame di pose1).
    I quat vanno sempre normalizzati qui: mju_mulPose non lo fa da solo, e i
    quat "as authored" nell'xml spesso non sono normalizzati (es. -1 1 0 0)."""
    quat1 = np.asarray(quat1, dtype=np.float64)
    quat1 = quat1 / np.linalg.norm(quat1)
    quat2 = np.asarray(quat2, dtype=np.float64)
    quat2 = quat2 / np.linalg.norm(quat2)
    pos_res = np.zeros(3)
    quat_res = np.zeros(4)
    mujoco.mju_mulPose(
        pos_res, quat_res,
        np.asarray(pos1, dtype=np.float64), quat1,
        np.asarray(pos2, dtype=np.float64), quat2,
    )
    return pos_res, quat_res


# rotazione mount(REP-103)->optical->convenzione camera MuJoCo, condivisa da
# tutte e 3 le RealSense (wrist, rs1, rs2): vedi discussione con Manuel,
# confermata contro la definizione ROS2 wrist_link->color_optical_frame
# rpy(-pi/2,0,-pi/2).
_Q_MECH_TO_OPTICAL = np.array([0.5, -0.5, 0.5, -0.5])
_Q_OPTICAL_TO_MJ = np.array([0.0, 1.0, 0.0, 0.0])
_, QUAT_ROS2MJ = _compose([0, 0, 0], _Q_MECH_TO_OPTICAL, [0, 0, 0], _Q_OPTICAL_TO_MJ)


def build_spec() -> mujoco.MjSpec:
    """Costruisce e ritorna lo MjSpec completo (fixed_parts + ur10e + 2f85 +
    camere + site open_tip/closed_tip/wrist_link), pronto per .compile().
    Nessun file scritto su disco: i meshdir dei child spec sono assoluti e
    risolti direttamente dal filesystem del pacchetto installato."""
    spec = mujoco.MjSpec.from_file(str(FIXED_PARTS_XML))

    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    arm = mujoco.MjSpec.from_file(str(UR10E_XML))
    arm.meshdir = str(UR10E_MESHDIR)
    mount_site = spec.site("ur10e_mount")
    mount_site.attach_body(arm.body("base"), "ur10e_", "")

    gripper = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    gripper.meshdir = str(GRIPPER_MESHDIR)
    tool_site = spec.site("ur10e_attachment_site")
    tool_site.attach_body(gripper.body("base_mount"), "2f85_", "")

    tool_pos = np.array(tool_site.pos)
    tool_quat = np.array(tool_site.quat)
    wrist3 = spec.body("ur10e_wrist_3_link")

    def add_relative_site(body, name, local_pos, local_quat, **kwargs):
        pos, quat = _compose(tool_pos, tool_quat, local_pos, local_quat)
        body.add_site(name=name, pos=pos, quat=quat, **kwargs)

    quat_rz_minus_90 = np.array([0.7071068, 0.0, 0.0, -0.7071068])
    add_relative_site(wrist3, "open_tip", local_pos=[0, 0, 0.149], local_quat=quat_rz_minus_90,
                       size=[0.006], rgba=[1, 1, 0, 1])
    add_relative_site(wrist3, "closed_tip", local_pos=[0, 0, 0.163], local_quat=quat_rz_minus_90,
                       size=[0.006], rgba=[1, 0.5, 0, 1])

    wrist_cam_quat = np.array([0.524378, 0.474031, -0.497434, 0.502874])
    add_relative_site(wrist3, "wrist_link", local_pos=[-0.0248083, -0.0607596, 0.0408365],
                       local_quat=wrist_cam_quat, size=[0.008], rgba=[0, 1, 1, 1])

    cam_pos, cam_quat = _compose(tool_pos, tool_quat,
                                  *_compose([-0.0248083, -0.0607596, 0.0408365],
                                            wrist_cam_quat, [0, 0, 0], QUAT_ROS2MJ))
    wrist3.add_camera(name="wrist_d435", pos=cam_pos, quat=cam_quat, fovy=42)

    cell = spec.body("cell_static")
    for cam_name in ["rs1", "rs2"]:
        site = spec.site(f"{cam_name}_camera_link_calibration_check")
        cam_pos, cam_quat = _compose(np.array(site.pos), np.array(site.quat), [0, 0, 0], QUAT_ROS2MJ)
        cell.add_camera(name=f"{cam_name}_d435", pos=cam_pos, quat=cam_quat, fovy=42)

    return spec


def build_model() -> mujoco.MjModel:
    """Costruisce e compila direttamente un MjModel in memoria (nessun file
    scritto su disco). E' la via usata da SharedworkCellSim."""
    return build_spec().compile()


def _fix_mesh_paths(xml_string: str) -> str:
    root = ET.fromstring(xml_string)
    compiler_el = root.find("compiler")
    if compiler_el is None:
        compiler_el = ET.SubElement(root, "compiler")
    compiler_el.set("meshdir", "assets")
    for mesh_el in root.iter("mesh"):
        name = mesh_el.get("name", "")
        file_attr = mesh_el.get("file")
        if file_attr is None:
            continue
        if name.startswith("ur10e_") and not file_attr.startswith("ur10e/"):
            mesh_el.set("file", "ur10e/" + file_attr)
        elif name.startswith("2f85_") and not file_attr.startswith("2f85/"):
            mesh_el.set("file", "2f85/" + file_attr)
    return ET.tostring(root, encoding="unicode")


def build_and_save_xml(output_xml_path: str, assets_dir_name: str = "assets") -> None:
    """Salva un MJCF standalone (+ una cartella `assets/` con le mesh
    accanto) utilizzabile con `simulate`, il viewer, o qualunque altro tool
    che sappia leggere un file .xml MuJoCo, senza dipendere dal pacchetto
    Python. Utile per il caso d'uso "standalone MuJoCo"."""
    import shutil

    spec = build_spec()
    spec.compile()  # valida prima di salvare
    xml_string = _fix_mesh_paths(spec.to_xml())

    out_path = Path(output_xml_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(xml_string)

    assets_out = out_path.parent / assets_dir_name
    for sub, src in [("ur10e", UR10E_MESHDIR), ("2f85", GRIPPER_MESHDIR)]:
        dst = assets_out / sub
        dst.mkdir(parents=True, exist_ok=True)
        for f in Path(src).iterdir():
            shutil.copy(f, dst / f.name)
