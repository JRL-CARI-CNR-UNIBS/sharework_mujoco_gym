"""
Uso standalone di SharedworkCellSim (nessuna dipendenza da gymnasium).

    python3 standalone_view.py                    # viewer interattivo
    python3 standalone_view.py --live              # viewer 3D + finestra live wrist camera (richiede opencv-python)
    python3 standalone_view.py --snapshot out.png  # salva un frame esterno
    python3 standalone_view.py --wrist-cam out.png # salva cosa "vede" la wrist_d435
"""
import argparse
import time

from sharework_mujoco.sim import SharedworkCellSim


def save_snapshot(sim, path, camera):
    import PIL.Image
    img = sim.render(camera=camera)
    PIL.Image.fromarray(img).save(path)
    print(f"Salvato {path} (camera={camera!r})")


def run_live_with_wrist_cam(sim):
    import cv2
    import mujoco.viewer

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            sim.step(sim.data.ctrl)  # mantiene ctrl correnti (impostabili dagli slider "Control")
            viewer.sync()

            img = sim.render(camera="wrist_d435")
            cv2.imshow("wrist_d435", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            dt = sim.model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--snapshot", metavar="OUT.png")
    parser.add_argument("--wrist-cam", metavar="OUT.png")
    args = parser.parse_args()

    sim = SharedworkCellSim()

    model, data = sim.model, sim.data
    import mujoco
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ur10e_attachment_site")
    tcp_pos = data.site_xpos[tcp_id]
    tcp_rot = data.site_xmat[tcp_id].reshape(3, 3)

    import numpy as np

    target = np.array([-0.03358524, -0.113, 1.3829712])  # sopra il tavolo, vedi fixed_parts.xml
    R_ref = np.array([
        [-2.05103490e-10, -1.00000000e+00, -2.05103533e-10],
        [-1.00000000e+00, 2.05103490e-10, 2.05103533e-10],
        [-2.05103533e-10, 2.05103533e-10, -1.00000000e+00]
    ])
    R_err = tcp_rot.T @ R_ref
    angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
    dist = np.linalg.norm(tcp_pos - target)
    print(f"tcp_pos: {tcp_pos},\n tcp_rot: {tcp_rot}")
    print(f"angle: {angle}, dist: {dist}")
    print(f"Caricato. nbody={sim.model.nbody} njnt={sim.model.njnt} nu={sim.model.nu}")

    if args.wrist_cam:
        save_snapshot(sim, args.wrist_cam, camera="wrist_d435")
        return
    if args.snapshot:
        save_snapshot(sim, args.snapshot, camera=None)
        return
    if args.live:
        run_live_with_wrist_cam(sim)
        return

    import mujoco.viewer
    print("Apro il viewer... (chiudi la finestra per uscire)")
    mujoco.viewer.launch(sim.model, sim.data)

    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ur10e_attachment_site")
    tcp_pos = data.site_xpos[tcp_id]
    tcp_rot = data.site_xmat[tcp_id].reshape(3, 3)

    import time
    time.sleep(10)
    tcp_pos = data.site_xpos[tcp_id]
    tcp_rot = data.site_xmat[tcp_id].reshape(3, 3)

    R_err = tcp_rot.T @ R_ref
    angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
    dist = np.linalg.norm(tcp_pos - target)
    print(f"tcp_pos: {tcp_pos},\n tcp_rot: {tcp_rot}")
    print(f"angle: {angle}, dist: {dist}")
    print(f"Caricato. nbody={sim.model.nbody} njnt={sim.model.njnt} nu={sim.model.nu}")


if __name__ == "__main__":
    main()
