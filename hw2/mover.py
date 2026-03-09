from pathlib import Path
import csv
import shutil

EXP_ROOT = Path("exp")
KEEP_DIR = EXP_ROOT / "keep"
LOG_GLOB = "InvertedPendulum*/log.csv"

TARGET_RETURN = 1000.0
MAX_STEPS = 100_000
EPS = 1e-9


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_run(log_path: Path) -> tuple[bool, bool]:
    """
    Returns (hit_before_100k_steps, hit_1000_ever).
    """
    hit_before_100k_steps = False
    hit_1000_ever = False

    with log_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {"Eval_MaxReturn", "Train_EnvstepsSoFar"}
        if not required.issubset(reader.fieldnames or []):
            return False, False

        for row in reader:
            max_return = parse_float(row.get("Eval_MaxReturn"))
            steps = parse_float(row.get("Train_EnvstepsSoFar"))

            if max_return is None or steps is None:
                continue

            if max_return >= TARGET_RETURN - EPS:
                hit_1000_ever = True
                if steps < MAX_STEPS:
                    hit_before_100k_steps = True
                    break

    return hit_before_100k_steps, hit_1000_ever


def unique_destination(dst: Path) -> Path:
    """Avoid overwriting if a same-name directory already exists in keep/."""
    if not dst.exists():
        return dst

    i = 1
    while True:
        candidate = dst.with_name(f"{dst.name}_{i}")
        if not candidate.exists():
            return candidate
        i += 1


def main():
    KEEP_DIR.mkdir(parents=True, exist_ok=True)

    log_files = sorted(EXP_ROOT.glob(LOG_GLOB))
    moved = []
    kept_in_place = []
    deleted = []

    for log_file in log_files:
        exp_dir = log_file.parent
        hit_before_100k_steps, hit_1000_ever = evaluate_run(log_file)

        if hit_before_100k_steps:
            dst = unique_destination(KEEP_DIR / exp_dir.name)
            shutil.move(str(exp_dir), str(dst))
            moved.append((exp_dir.name, dst.name))
        elif not hit_1000_ever:
            shutil.rmtree(exp_dir)
            deleted.append(exp_dir.name)
        else:
            kept_in_place.append(exp_dir.name)

    print(f"Found {len(log_files)} experiments")
    print(f"Moved to keep/: {len(moved)}")
    print(f"Directories moved: {[src for src, _ in moved]}")
    print(f"Deleted from exp/: {len(deleted)}")
    print(f"Directories deleted: {deleted}")
    print(f"Left in exp/: {len(kept_in_place)}")

    if moved:
        print("\nMoved experiments:")
        for src_name, dst_name in moved:
            print(f"  {src_name} -> keep/{dst_name}")


if __name__ == "__main__":
    main()