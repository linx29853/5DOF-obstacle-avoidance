"""单进程 APF / 单树 RRT / RRT-Connect 固定场景对比。"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from so101.benchmark import CONFIG, measure, metadata, problem_for
from so101.benchmark_resources import enable_guard
from so101.planners import Budget, Options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    enable_guard()
    suite = json.loads(CONFIG.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metadata.json").write_text(json.dumps(metadata(CONFIG), indent=2))
    runs = [("apf", 0)] + [(p, s) for p in ["rrt", "rrt_connect"] for s in range(3)]
    total = len(suite["scenes"]) * len(runs)
    rows = []
    with (args.output / "records.jsonl").open("w") as stream:
        for scene in suite["scenes"]:
            query = next(q for q in scene["queries"] if q["split"] == "formal")
            problem = problem_for(scene["scene"], query)
            for planner, seed in runs:
                row = measure(
                    problem, planner, Budget(seconds=args.seconds), seed, Options()
                )
                row.update(scene=scene["id"], query=query["id"])
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                rows.append(row)
                print(
                    f"{len(rows)}/{total} {row['scene']} {planner} seed={seed}: {row['reason']}",
                    flush=True,
                )
    fields = [
        "scene",
        "planner",
        "seed",
        "success",
        "reason",
        "first_solution_seconds",
        "length_rad",
        "simplified_length_rad",
    ]
    with (args.output / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    from so101.apf_report import generate

    generate(args.output)


if __name__ == "__main__":
    main()
