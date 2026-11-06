"""固定 SO-101 桌面搬运、隔挡转移和分拣演示。"""

import argparse

from so101.manipulation import DEFAULT_CONFIG, play_task, run_task

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=["tabletop", "barrier", "sorting", "detour", "clutter"],
        default="tabletop",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--export", action="store_true", help="导出 GIF、封面及轨迹图")
    args = parser.parse_args()
    result = run_task(args.case, args.config, args.output_dir)
    if args.export:
        from so101.manipulation_media import export_task, plot_task

        export_task(result)
        plot_task(result)
    print("结果目录:", result)
    if not args.check_only:
        play_task(result)
