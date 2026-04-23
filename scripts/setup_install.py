#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV_DIR = REPO_ROOT / ".venv"
DEFAULT_SAM3_VENV_DIR = REPO_ROOT / ".venv-sam3"
DEFAULT_SAM3_REPO = REPO_ROOT / "extern" / "sam3"
DEFAULT_HUNYUAN_REPO = REPO_ROOT / "extern" / "Hunyuan3Dv2"
DEFAULT_FTETWILD_REPO = REPO_ROOT / "extern" / "fTetWild"
DEFAULT_FTETWILD_BUILD_DIR = REPO_ROOT / "extern" / "fTetWild" / "build-local"
DEFAULT_DEPTH_PRO_CHECKPOINT = REPO_ROOT / "checkpoints" / "depth_pro.pt"
DEFAULT_SAM3_CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "sam3"
DEFAULT_HF_TOKEN_PATH = Path.home() / ".cache" / "huggingface" / "token"
DEFAULT_ENV_OUTPUT = REPO_ROOT / ".env.install"
DEFAULT_PRIVATE_WHEEL_DIR = REPO_ROOT / "private_wheels"
HUNYUAN_REPO_URL = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git"
HUNYUAN_REF = "f8db63096c8282cb27354314d896feba5ba6ff8a"
FTETWILD_REPO_URL = "https://github.com/wildmeshing/fTetWild.git"
SYSTEM_PACKAGES = [
    "build-essential",
    "ca-certificates",
    "cmake",
    "curl",
    "git",
    "gnupg",
    "jq",
    "libegl1",
    "libgbm1",
    "libgl1",
    "libglib2.0-0",
    "libgmp-dev",
    "libopengl0",
    "libsm6",
    "libtbb12",
    "libxext6",
    "libxrender1",
    "ninja-build",
]
PYTHON_VERSION = "3.10"
NODE_MAJOR = 20
DEPTH_PRO_URL = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"


def log(message: str) -> None:
    print(f"[install] {message}", flush=True)


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def load_install_env() -> dict[str, str]:
    merged: dict[str, str] = {}
    for env_path in (REPO_ROOT / ".env", Path.home() / ".env"):
        merged.update(parse_env_file(env_path))
    merged.update(os.environ)
    return merged


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    rendered = " ".join(shlex.quote(part) for part in command)
    log(f"run: {rendered}")
    subprocess.run(command, cwd=cwd, env=env, check=True)


def capture(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def run_shell(script: str, *, env: dict[str, str], use_root: bool = False) -> None:
    command: list[str]
    if use_root:
        if os.geteuid() == 0:
            command = ["bash", "-lc", script]
        elif shutil.which("sudo"):
            command = ["sudo", "bash", "-lc", script]
        else:
            raise SystemExit("System package installation requires root or sudo.")
    else:
        command = ["bash", "-lc", script]
    run(command, cwd=REPO_ROOT, env=env)


def resolve_torch_cuda_arch_list(env: dict[str, str]) -> str:
    configured = str(env.get("PAT3D_TORCH_CUDA_ARCH_LIST") or env.get("TORCH_CUDA_ARCH_LIST") or "").strip()
    if configured:
        return configured
    try:
        raw = capture(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            env=env,
        )
    except Exception:
        raw = ""
    values: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        normalized = line.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)
    if values:
        return ";".join(values)
    return "12.0"


def ensure_supported_environment() -> None:
    if platform.system() != "Linux":
        raise SystemExit("PAT3D install is only supported on Linux hosts.")
    fs_type = capture(["stat", "-f", "-c", "%T", str(REPO_ROOT)])
    if fs_type.lower() in {"cifs", "smb2", "smb3"}:
        raise SystemExit(
            "PAT3D install is not supported from a CIFS/SMB mount. "
            "Move the repo to a local Linux filesystem such as ext4."
        )


def ensure_uv(env: dict[str, str]) -> None:
    if shutil.which("uv", path=env.get("PATH")):
        return
    log("installing uv")
    run_shell("curl -LsSf https://astral.sh/uv/install.sh | sh", env=env)
    env["PATH"] = f"{Path.home() / '.local' / 'bin'}:{env.get('PATH', '')}"
    if not shutil.which("uv", path=env.get("PATH")):
        raise SystemExit("uv installation completed, but `uv` is still not on PATH.")


def ensure_system_dependencies(env: dict[str, str], *, skip_system_deps: bool) -> None:
    if skip_system_deps:
        return
    if not shutil.which("apt-get", path=env.get("PATH")):
        raise SystemExit(
            "Automatic system dependency installation currently supports Ubuntu/Debian only. "
            "Re-run with --skip-system-deps after installing the required packages manually."
        )
    node_major = 0
    if shutil.which("node", path=env.get("PATH")):
        try:
            node_major = int(capture(["node", "--version"]).lstrip("v").split(".", 1)[0])
        except Exception:
            node_major = 0
    if node_major < NODE_MAJOR:
        log(f"installing Node.js {NODE_MAJOR} from NodeSource")
        run_shell(
            """
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl gnupg
mkdir -p /etc/apt/keyrings
curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list
apt-get update
apt-get install -y nodejs
""",
            env=env,
            use_root=True,
        )
    package_list = " ".join(SYSTEM_PACKAGES)
    log("installing supported Ubuntu system packages")
    run_shell(
        f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y {package_list}
""",
        env=env,
        use_root=True,
    )


def ensure_hf_token_file(env: dict[str, str], configured_path: Path) -> Path:
    if configured_path.exists():
        return configured_path
    token = (
        env.get("HF_TOKEN")
        or env.get("HUGGINGFACE_HUB_TOKEN")
        or env.get("HUGGING_FACE_HUB_TOKEN")
        or ""
    ).strip()
    if not token:
        raise SystemExit(
            "A Hugging Face token with access to `facebook/sam3` is required. "
            "Set HF_TOKEN in .env or create ~/.cache/huggingface/token."
        )
    configured_path.parent.mkdir(parents=True, exist_ok=True)
    configured_path.write_text(token + "\n", encoding="utf-8")
    return configured_path


def ensure_git_checkout(
    env: dict[str, str],
    destination: Path,
    *,
    repo_url: str,
    ref: str | None = None,
) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    log(f"cloning {repo_url} into {destination}")
    run(["git", "clone", "--filter=blob:none", repo_url, str(destination)], env=env)
    if ref:
        run(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", ref], env=env)
        run(["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"], env=env)


def validate_hf_access(env: dict[str, str], venv_python: Path, token_path: Path) -> None:
    log("validating Hugging Face access to facebook/sam3")
    run(
        [
            str(venv_python),
            "-c",
            (
                "from pathlib import Path; "
                "from huggingface_hub import hf_hub_download; "
                "token = Path(__import__('sys').argv[1]).read_text(encoding='utf-8').strip(); "
                "assert token, 'empty Hugging Face token'; "
                "hf_hub_download(repo_id='facebook/sam3', filename='config.json', token=token); "
                "print('sam3_hf_access_ok')"
            ),
            str(token_path),
        ],
        env=env,
    )


def resolve_bundled_private_wheel() -> Path | None:
    candidates = sorted(DEFAULT_PRIVATE_WHEEL_DIR.glob("pyuipc-*.whl"))
    if not candidates:
        return None
    return candidates[-1]


def resolve_private_wheel(env: dict[str, str], venv_python: Path, provided: str | None) -> Path:
    wheel_value = str(provided or "").strip()
    if wheel_value:
        wheel_path = Path(wheel_value).expanduser().resolve()
    else:
        wheel_path = resolve_bundled_private_wheel()
        if wheel_path is None:
            raise SystemExit(
                "PAT3D requires a Diff_GIPC wheel. Place the tracked prebuilt wheel under "
                f"{DEFAULT_PRIVATE_WHEEL_DIR} or set PAT3D_PRIVATE_PHYSICS_WHEEL / "
                "--private-physics-wheel to an override path."
            )
        log(f"using bundled private physics wheel: {wheel_path}")
    if not wheel_path.is_file():
        raise SystemExit(f"Private physics wheel does not exist: {wheel_path}")
    return wheel_path


def install_python_environment(env: dict[str, str], venv_dir: Path) -> Path:
    log(f"creating uv venv at {venv_dir}")
    run(["uv", "python", "install", PYTHON_VERSION], env=env)
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    run(["uv", "venv", "--python", PYTHON_VERSION, str(venv_dir)], env=env)
    return venv_dir / "bin" / "python"


def ensure_hunyuan_checkout(env: dict[str, str], hunyuan_repo: Path) -> None:
    ensure_git_checkout(
        env,
        hunyuan_repo,
        repo_url=HUNYUAN_REPO_URL,
        ref=HUNYUAN_REF,
    )


def ensure_ftetwild_checkout(env: dict[str, str], ftetwild_repo: Path) -> None:
    ensure_git_checkout(
        env,
        ftetwild_repo,
        repo_url=FTETWILD_REPO_URL,
    )


def install_python_dependencies(env: dict[str, str], venv_python: Path, hunyuan_repo: Path) -> None:
    run(
        [
            "uv",
            "pip",
            "install",
            "--index-strategy",
            "unsafe-best-match",
            "--python",
            str(venv_python),
            "-r",
            "requirements.txt",
        ],
        cwd=REPO_ROOT,
        env=env,
    )
    run(
        [
            "uv",
            "pip",
            "install",
            "--index-strategy",
            "unsafe-best-match",
            "--python",
            str(venv_python),
            "-e",
            str(hunyuan_repo),
        ],
        cwd=REPO_ROOT,
        env=env,
    )
    run(
        [
            "uv",
            "pip",
            "install",
            "--index-strategy",
            "unsafe-best-match",
            "--python",
            str(venv_python),
            ".",
        ],
        cwd=REPO_ROOT,
        env=env,
    )


def validate_core_imports(env: dict[str, str], venv_python: Path) -> None:
    run(
        [
            str(venv_python),
            "-c",
            "import pat3d; print('pat3d', pat3d.__version__)",
        ],
        cwd=REPO_ROOT,
        env=env,
    )


def install_private_wheel(env: dict[str, str], venv_python: Path, wheel_path: Path) -> None:
    run(
        [
            "uv",
            "pip",
            "install",
            "--index-strategy",
            "unsafe-best-match",
            "--python",
            str(venv_python),
            str(wheel_path),
        ],
        cwd=REPO_ROOT,
        env=env,
    )
    run(
        [
            str(venv_python),
            "-c",
            (
                "import importlib; "
                "mods=('uipc','uipc.core','uipc.geometry','uipc.constitution','uipc.gui','uipc.torch','pyuipc'); "
                "[print(name, 'ok', getattr(importlib.import_module(name), '__file__', None)) for name in mods]"
            ),
        ],
        cwd=REPO_ROOT,
        env=env,
    )


def ensure_depth_pro_checkpoint(env: dict[str, str], checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        return
    run(["curl", "-L", "--fail", DEPTH_PRO_URL, "-o", str(checkpoint_path)], env=env)


def build_ftetwild(env: dict[str, str], source_dir: Path, build_dir: Path) -> Path:
    run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=REPO_ROOT,
        env=env,
    )
    run(
        ["cmake", "--build", str(build_dir), "-j", str(os.cpu_count() or 1)],
        cwd=REPO_ROOT,
        env=env,
    )
    binary_path = build_dir / "FloatTetwild_bin"
    if not binary_path.is_file():
        raise SystemExit(f"Missing FloatTetWild binary: {binary_path}")
    return binary_path


def build_hunyuan_extensions(env: dict[str, str], venv_python: Path, hunyuan_repo: Path) -> None:
    build_env = dict(env)
    build_env.setdefault("TORCH_CUDA_ARCH_LIST", resolve_torch_cuda_arch_list(env))
    run(
        [str(venv_python), "setup.py", "install"],
        cwd=hunyuan_repo / "hy3dgen" / "texgen" / "custom_rasterizer",
        env=build_env,
    )
    run(
        [str(venv_python), "setup.py", "install"],
        cwd=hunyuan_repo / "hy3dgen" / "texgen" / "differentiable_renderer",
        env=build_env,
    )
    run(
        [
            str(venv_python),
            "-c",
            (
                "import importlib; "
                "mods=('depth_pro','plyfile','open3d','pymeshlab','hy3dgen','custom_rasterizer','custom_rasterizer_kernel','mesh_processor'); "
                "[print(name, 'ok', getattr(importlib.import_module(name), '__file__', None)) for name in mods]"
            ),
        ],
        cwd=REPO_ROOT,
        env=build_env,
    )


def ensure_sam3_checkout(env: dict[str, str], sam3_repo: Path) -> None:
    ensure_git_checkout(
        env,
        sam3_repo,
        repo_url="https://github.com/facebookresearch/sam3",
    )


def install_sam3_environment(env: dict[str, str], sam3_repo: Path, sam3_venv_dir: Path, token_path: Path, checkpoint_dir: Path) -> Path:
    ensure_sam3_checkout(env, sam3_repo)
    if sam3_venv_dir.exists():
        shutil.rmtree(sam3_venv_dir)
    run(["uv", "venv", "--python", PYTHON_VERSION, str(sam3_venv_dir)], cwd=REPO_ROOT, env=env)
    sam3_python = sam3_venv_dir / "bin" / "python"
    run(
        [
            "uv",
            "pip",
            "install",
            "--index-strategy",
            "unsafe-best-match",
            "--python",
            str(sam3_python),
            "torch==2.13.0.dev20260421+cu130",
            "torchvision==0.27.0.dev20260421+cu130",
            "torchaudio==2.11.0.dev20260421+cu130",
            "--extra-index-url",
            "https://download.pytorch.org/whl/nightly/cu130",
        ],
        cwd=REPO_ROOT,
        env=env,
    )
    run(
        [
            "uv",
            "pip",
            "install",
            "--index-strategy",
            "unsafe-best-match",
            "--python",
            str(sam3_python),
            "-e",
            str(sam3_repo),
            "decord",
            "pycocotools",
            "einops",
            "psutil",
        ],
        cwd=REPO_ROOT,
        env=env,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(sam3_python),
            "-c",
            (
                "from pathlib import Path; import shutil, sys; "
                "from huggingface_hub import hf_hub_download; "
                "token = Path(sys.argv[1]).read_text(encoding='utf-8').strip(); "
                "output_dir = Path(sys.argv[2]); output_dir.mkdir(parents=True, exist_ok=True); "
                "files=('config.json','sam3.pt'); "
                "[shutil.copy2(Path(hf_hub_download(repo_id='facebook/sam3', filename=name, token=token)), output_dir / name) for name in files]; "
                "print('sam3_ready', output_dir)"
            ),
            str(token_path),
            str(checkpoint_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
    )
    run(
        [
            str(sam3_python),
            "-c",
            (
                "from pathlib import Path; "
                "import sam3, torch; "
                "from sam3.model.sam3_image_processor import Sam3Processor; "
                "checkpoint = Path(__import__('sys').argv[1]); "
                "assert checkpoint.is_file(), f'missing SAM3 checkpoint: {checkpoint}'; "
                "print('sam3', sam3.__file__, 'cuda', torch.cuda.is_available())"
            ),
            str(checkpoint_dir / "sam3.pt"),
        ],
        cwd=REPO_ROOT,
        env=env,
    )
    return sam3_python


def ensure_dashboard_dependencies(env: dict[str, str], *, install_playwright_browser: bool) -> None:
    run(["npm", "install"], cwd=REPO_ROOT / "dashboard", env=env)
    if install_playwright_browser:
        run(["npx", "playwright", "install", "--with-deps", "chromium"], cwd=REPO_ROOT / "dashboard", env=env)


def write_launch_env(
    env_output_path: Path,
    *,
    venv_python: Path,
    sam3_python: Path,
    sam3_checkpoint: Path,
    ftetwild_bin: Path,
    private_wheel: Path,
) -> None:
    env_output_path.write_text(
        "\n".join(
            [
                "# Generated by scripts/setup_install.py",
                f"PAT3D_DASHBOARD_PYTHON={venv_python}",
                f"PAT3D_LEGACY_PHYSICS_PYTHON={venv_python}",
                f"PAT3D_SAM3_PYTHON={sam3_python}",
                f"PAT3D_SAM3_CHECKPOINT_PATH={sam3_checkpoint}",
                "PAT3D_SAM3_LOAD_FROM_HF=0",
                f"PAT3D_FTETWILD_BIN={ftetwild_bin}",
                f"PAT3D_PRIVATE_PHYSICS_WHEEL={private_wheel}",
                "PAT3D_DASHBOARD_STAGE_BACKENDS_PROFILE=default",
                "PAT3D_DASHBOARD_FORCE_STAGE_BACKENDS_PROFILE=1",
                f"PAT3D_REPO_ROOT={REPO_ROOT}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Provision the supported local PAT3D install.")
    parser.add_argument("--private-physics-wheel")
    parser.add_argument("--venv-dir", default=str(DEFAULT_VENV_DIR))
    parser.add_argument("--sam3-venv-dir", default=str(DEFAULT_SAM3_VENV_DIR))
    parser.add_argument("--sam3-repo", default=str(DEFAULT_SAM3_REPO))
    parser.add_argument("--hunyuan-repo", default=str(DEFAULT_HUNYUAN_REPO))
    parser.add_argument("--ftetwild-repo", default=str(DEFAULT_FTETWILD_REPO))
    parser.add_argument("--ftetwild-build-dir", default=str(DEFAULT_FTETWILD_BUILD_DIR))
    parser.add_argument("--depth-pro-checkpoint", default=str(DEFAULT_DEPTH_PRO_CHECKPOINT))
    parser.add_argument("--hf-token-file")
    parser.add_argument("--env-output", default=str(DEFAULT_ENV_OUTPUT))
    parser.add_argument("--skip-system-deps", action="store_true")
    parser.add_argument("--skip-playwright-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = load_install_env()
    env["PATH"] = f"{Path.home() / '.local' / 'bin'}:{env.get('PATH', '')}"
    ensure_supported_environment()
    ensure_system_dependencies(env, skip_system_deps=args.skip_system_deps)
    ensure_uv(env)

    venv_dir = Path(args.venv_dir).expanduser().resolve()
    sam3_venv_dir = Path(args.sam3_venv_dir).expanduser().resolve()
    sam3_repo = Path(args.sam3_repo).expanduser().resolve()
    hunyuan_repo = Path(args.hunyuan_repo).expanduser().resolve()
    ftetwild_repo = Path(args.ftetwild_repo).expanduser().resolve()
    ftetwild_build_dir = Path(args.ftetwild_build_dir).expanduser().resolve()
    depth_pro_checkpoint = Path(args.depth_pro_checkpoint).expanduser().resolve()
    hf_token_path_value = args.hf_token_file or env.get("PAT3D_HF_TOKEN_FILE") or str(DEFAULT_HF_TOKEN_PATH)
    hf_token_path = ensure_hf_token_file(env, Path(hf_token_path_value).expanduser().resolve())
    env_output_path = Path(args.env_output).expanduser().resolve()
    private_physics_wheel_arg = args.private_physics_wheel or env.get("PAT3D_PRIVATE_PHYSICS_WHEEL")

    ensure_hunyuan_checkout(env, hunyuan_repo)
    ensure_ftetwild_checkout(env, ftetwild_repo)
    venv_python = install_python_environment(env, venv_dir)
    install_python_dependencies(env, venv_python, hunyuan_repo)
    validate_core_imports(env, venv_python)
    validate_hf_access(env, venv_python, hf_token_path)
    private_wheel = resolve_private_wheel(env, venv_python, private_physics_wheel_arg)
    install_private_wheel(env, venv_python, private_wheel)
    ensure_depth_pro_checkpoint(env, depth_pro_checkpoint)
    ftetwild_bin = build_ftetwild(env, ftetwild_repo, ftetwild_build_dir)
    build_hunyuan_extensions(env, venv_python, hunyuan_repo)
    sam3_python = install_sam3_environment(
        env,
        sam3_repo,
        sam3_venv_dir,
        hf_token_path,
        DEFAULT_SAM3_CHECKPOINT_DIR.resolve(),
    )
    ensure_dashboard_dependencies(env, install_playwright_browser=not args.skip_playwright_browser)
    write_launch_env(
        env_output_path,
        venv_python=venv_python,
        sam3_python=sam3_python,
        sam3_checkpoint=DEFAULT_SAM3_CHECKPOINT_DIR.resolve() / "sam3.pt",
        ftetwild_bin=ftetwild_bin,
        private_wheel=private_wheel,
    )

    log("install complete")
    print(f"launch env file: {env_output_path}")
    print(f"dashboard python: {venv_python}")
    print(f"sam3 python: {sam3_python}")
    print(f"FloatTetWild: {ftetwild_bin}")
    print(f"Depth Pro checkpoint: {depth_pro_checkpoint}")
    print(f"SAM3 checkpoint: {DEFAULT_SAM3_CHECKPOINT_DIR.resolve() / 'sam3.pt'}")
    print()
    print("Next:")
    if Path("/.dockerenv").exists():
        print("  bash scripts/run_docker_dashboard.sh")
    else:
        print("  bash scripts/run_dashboard.sh")


if __name__ == "__main__":
    main()
