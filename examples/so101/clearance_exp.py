"""展示未碰撞但间隙不足的姿态，验证安全裕量判断。"""

import argparse

import numpy as np

from so101.collision import has_clearance, is_pose_valid
from so101.collision_scene import inspect_contacts, load_collision_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--margin-mm", type=float, default=5.0)
    args = parser.parse_args()
    margin = args.margin_mm / 1000.0
    model, config, joint_ids = load_collision_scene(query_margin=margin)
    unfinished = False
    for base_angle in [-60, -40, -38, -36, 0]:
        angles = np.deg2rad([base_angle, 0, 0, 0, 0])
        _, contacts = inspect_contacts(model, config, joint_ids, angles)
        original_valid = is_pose_valid(angles, config["joint_limits_rad"], contacts)
        print(f"\n底座角 {base_angle}°；原碰撞判断: {original_valid}")
        if contacts:
            print(f"查询范围内最小间隙: {min(contact['dist'] for contact in contacts)*1000:.3f} mm")
        else:
            print(f"查询范围内没有近距离几何体（查询范围 {args.margin_mm:g} mm）")
        try:
            clearance_valid = has_clearance(contacts, margin)
            print(f"满足 {args.margin_mm:g} mm 安全裕量:", original_valid and clearance_valid)
        except NotImplementedError:
            unfinished = True
    if unfinished:
        print("\n请填写 src/so101/collision.py 中的 has_clearance，然后重新运行。")


if __name__ == "__main__":
    main()
