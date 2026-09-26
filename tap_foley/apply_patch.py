"""Apply the matching patch to a user-supplied upstream checkout."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("mmaudio", "omni2sound"))
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--check", action="store_true", help="Validate without changing files")
    args = parser.parse_args()
    patch_root = Path(__file__).resolve().parents[1] / "patches"
    if not (patch_root / "upstream.json").is_file():
        parser.error("Run from an editable TAP-Foley checkout: pip install -e . --no-deps")
    config = json.loads((patch_root / "upstream.json").read_text())[args.backend]
    patch = patch_root / config["patch"]
    command = ["git", "-C", str(args.checkout), "apply"]
    if subprocess.run(command + ["--reverse", "--check", str(patch)], capture_output=True).returncode == 0:
        print("TAP-Foley patch is already applied")
        return
    result = subprocess.run(command + ["--check", str(patch)], capture_output=True, text=True)
    if result.returncode:
        parser.error(f"Patch does not apply; use upstream {config['commit']}.\n{result.stderr}")
    if not args.check:
        subprocess.run(command + [str(patch)], check=True)
    print("TAP-Foley patch " + ("validated" if args.check else "applied"))


if __name__ == "__main__":
    main()
