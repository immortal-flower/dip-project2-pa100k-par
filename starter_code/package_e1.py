"""Bundle repository source/evidence with file hashes, excluding ignored data."""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    paths = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT)
    names = sorted(set(name.decode("utf-8") for name in paths.split(b"\0") if name))
    manifest = {"base_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
                "note": "Local A-line source snapshot, not a new remote commit; no dataset/checkpoint included.", "sha256": {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            path = (ROOT / name).resolve()
            path.relative_to(ROOT)
            if not path.is_file():
                continue
            if path.suffix.lower() in (".pt", ".pth", ".jpg", ".jpeg", ".npz", ".csv", ".mat"):
                raise ValueError(f"Unexpected data/model artifact in source list: {name}")
            content = path.read_bytes()
            manifest["sha256"][name] = hashlib.sha256(content).hexdigest()
            archive.writestr(name, content)
        archive.writestr("SOURCE_MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
    print(f"Saved {output}; {len(manifest['sha256'])} files; SHA-256 {hashlib.sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
