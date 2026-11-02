"""一键运行 SO-101 避障 Demo。"""

import argparse

from so101.demo import DEFAULT_CONFIG, run_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Demo 配置文件")
    parser.add_argument("--case", choices=("single", "multi", "blocked"), default="multi")
    parser.add_argument("--check-only", action="store_true", help="完成全部计算与复核，不打开窗口")
    args = parser.parse_args()
    try:
        report = run_demo(args.config, args.case, show_viewer=not args.check_only)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(2, f"Demo 未完成：{error}\n")
    if report["status"] != "success":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
