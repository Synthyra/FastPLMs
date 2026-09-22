"""Drive confidence-head stages on a remote GPU workstation over SSH.

Stages run on the workstation. This module copies source files, delivers the Hugging Face and
Weights & Biases credentials without printing them, starts detached tmux jobs that log to
`~/data/confidence-v2/logs/<job>.log`, and waits for a job to exit.
"""

from __future__ import annotations

import argparse
import io
import re
import shlex
import subprocess
import tarfile

from pathlib import Path

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[2]
REMOTE_REPO = "work/FastPLMs"  # relative to the remote home directory
REMOTE_PYTHON = "~/venvs/fastplms/bin/python"
REMOTE_LOGS = "~/data/confidence-v2/logs"
SYNCED_PREFIXES = (
    "src/",
    "tools/",
    "tests/",
    "requirements/",
    "pytest.ini",
    "ruff.toml",
    "mypy.ini",
)
SENSITIVE_SUFFIXES = (".env", ".pem", ".key", ".p12", ".pfx")
SECRET_NAMES = ("HF_TOKEN", "WANDB_API_KEY")


def validate_job_name(job: str) -> str:
    """Keep tmux names and remote log paths to one shell-safe component."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", job) is None:
        raise ValueError(
            "Job names need 1-80 letters, digits, underscores or hyphens, "
            "starting with a letter or digit"
        )
    return job


def ssh_command(host: str, identity: Path, remote_command: str) -> list[str]:
    return [
        "ssh",
        "-i",
        str(identity),
        "-o",
        "BatchMode=yes",
        "-o",
        "ServerAliveInterval=60",
        host,
        remote_command,
    ]


def synced_files() -> list[str]:
    """Tracked and untracked, non-ignored files under the synced prefixes."""
    listing = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode()
    return sorted(
        name
        for name in listing.split("\0")
        if name.startswith(SYNCED_PREFIXES)
        and not name.endswith(SENSITIVE_SUFFIXES)
        and ".secrets" not in name
        and (ROOT / name).is_file()
    )


def sync(host: str, identity: Path) -> None:
    buffer = io.BytesIO()
    names = synced_files()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name in names:
            tar.add(ROOT / name, arcname=name)
    # Replace synced directories so deleted local files do not linger remotely.
    directories = " ".join(sorted({name.split("/", 1)[0] for name in names if "/" in name}))
    command = f"mkdir -p {REMOTE_REPO} && cd {REMOTE_REPO} && rm -rf {directories} && tar -xzf -"
    subprocess.run(ssh_command(host, identity, command), input=buffer.getvalue(), check=True)
    print(f"synced {len(names)} files to {host}:{REMOTE_REPO}")


def push_secrets(host: str, identity: Path) -> None:
    """Write only the named credentials into the remote repository's `.secrets.env`."""
    values = dotenv_values(ROOT / ".secrets.env")
    missing = [name for name in SECRET_NAMES if not values.get(name)]
    if missing:
        raise RuntimeError(f"missing credentials in .secrets.env: {missing}")
    content = "".join(f"{name}={values[name]}\n" for name in SECRET_NAMES).encode()
    command = f"umask 077 && mkdir -p {REMOTE_REPO} && cat > {REMOTE_REPO}/.secrets.env"
    subprocess.run(ssh_command(host, identity, command), input=content, check=True)
    print(f"delivered {', '.join(SECRET_NAMES)} to {host}:{REMOTE_REPO}/.secrets.env")


def start(
    host: str, identity: Path, job: str, stage_arguments: list[str], after: str | None
) -> None:
    """Start `python -m tools.confidence.host <stage_arguments>` in a detached tmux session.

    With `after`, the stage waits until that job's log ends with its exit status and runs only if
    the status is 0, so long stages can be queued behind one another.
    """
    validate_job_name(job)
    if after is not None:
        validate_job_name(after)
    stage = " ".join(shlex.quote(argument) for argument in stage_arguments)
    run_stage = (
        f"{{ PYTHONPATH=src:. {REMOTE_PYTHON} -X utf8 -m tools.confidence.host {stage}; "
        f'echo "[exit status $?]"; }} 2>&1 | tee -a {REMOTE_LOGS}/{job}.log'
    )
    if after is None:
        job_command = f"cd ~/{REMOTE_REPO} && {run_stage}"
    else:
        previous_log = f"{REMOTE_LOGS}/{after}.log"
        job_command = (
            f"cd ~/{REMOTE_REPO} && until tail -n 1 {previous_log} 2>/dev/null | "
            "grep -q '^\\[exit status'; do sleep 60; done; "
            f"if tail -n 1 {previous_log} | grep -q '^\\[exit status 0\\]'; then {run_stage}; "
            f'else echo "[exit status 1] {after} did not exit cleanly" | '
            f"tee -a {REMOTE_LOGS}/{job}.log; fi"
        )
    command = (
        f"mkdir -p {REMOTE_LOGS} && tmux new-session -d -s {shlex.quote(job)} "
        f"{shlex.quote('bash -o pipefail -c ' + shlex.quote(job_command))}"
    )
    subprocess.run(ssh_command(host, identity, command), check=True)
    print(f"started tmux session {job} on {host}")


def wait(host: str, identity: Path, job: str, tail_bytes: int) -> None:
    """Wait for the recorded job status and print its log tail without progress bars."""
    validate_job_name(job)
    if tail_bytes <= 0:
        raise ValueError("tail_bytes must be positive")
    log_path = f"{REMOTE_LOGS}/{job}.log"
    command = (
        "status=''; while [ -z \"$status\" ]; do "
        f"status=$(tail -n 1 {log_path} 2>/dev/null | "
        "sed -n 's/^\\[exit status \\([0-9][0-9]*\\)\\].*$/\\1/p'); "
        '[ -n "$status" ] || sleep 15; done; '
        f"grep -a -v -e 'Loading weights' -e 'it/s]' {log_path} | tail -c {tail_bytes}; "
        'exit "$status"'
    )
    subprocess.run(ssh_command(host, identity, command), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", required=True, help="user@address of the workstation")
    parser.add_argument("--identity", required=True, type=Path, help="SSH private key path")
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("sync", help="copy source files to the workstation")
    actions.add_parser(
        "push-secrets", help="deliver HF_TOKEN and WANDB_API_KEY without printing them"
    )
    start_parser = actions.add_parser("start", help="run a host stage in a detached tmux job")
    start_parser.add_argument(
        "--after", help="run only after this job exits with status 0; give it before the job name"
    )
    start_parser.add_argument("job", help="tmux session and log name")
    start_parser.add_argument("stage_arguments", nargs=argparse.REMAINDER)
    wait_parser = actions.add_parser("wait", help="wait for a job to exit and print its log tail")
    wait_parser.add_argument("job", help="tmux session and log name")
    wait_parser.add_argument("--tail-bytes", type=int, default=6000)
    args = parser.parse_args()

    if args.action == "sync":
        sync(args.host, args.identity)
    elif args.action == "push-secrets":
        push_secrets(args.host, args.identity)
    elif args.action == "start":
        start(args.host, args.identity, args.job, args.stage_arguments, args.after)
    elif args.action == "wait":
        try:
            wait(args.host, args.identity, args.job, args.tail_bytes)
        except subprocess.CalledProcessError as error:
            raise SystemExit(error.returncode) from None


if __name__ == "__main__":
    main()
