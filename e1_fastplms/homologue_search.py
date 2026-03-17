"""
Homologue retrieval via MMseqs2 (Docker) or the ColabFold API.
Produces .a3m MSA files compatible with the RAG-E1 pipeline.
"""

import os
import hashlib
import shutil
import subprocess
import logging
import time
import random
import tarfile
from pathlib import Path
from typing import Optional

import psutil
import requests
import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

DOCKER_IMAGE = "ghcr.io/soedinglab/mmseqs2"
COLABFOLD_HOST = "https://api.colabfold.com"


class HomologueSearcher:
    """MMseqs2-based homologue search via Docker."""

    def __init__(
        self,
        target_db: str,
        docker_image: str = DOCKER_IMAGE,
        sensitivity: float = 7.5,
        max_seqs: int = 1000,
        min_seq_id: float = 0.0,
        coverage: float = 0.8,
        memory_fraction: float = 0.5,
    ):
        self.target_db = target_db
        self.docker_image = docker_image
        self.sensitivity = sensitivity
        self.max_seqs = max_seqs
        self.min_seq_id = min_seq_id
        self.coverage = coverage
        self.memory_fraction = memory_fraction


    def _run_docker_command(self, cmd: list, **kwargs):
        try:
            result = subprocess.run(cmd, **kwargs)
            if result.returncode == 0:
                return result
        except Exception:
            pass

        if os.name == "nt":
            return subprocess.run(cmd, **kwargs)

        return subprocess.run(["sudo"] + cmd, **kwargs)

    def _validate_paths_under_cwd(self, *paths: str) -> None:
        cwd = os.path.abspath(os.getcwd())
        for p in paths:
            ap = os.path.abspath(p)
            if not (ap == cwd or ap.startswith(cwd + os.sep)):
                raise ValueError(
                    "Path must be under current working directory for docker volume mount. "
                    f"cwd={cwd!r}, path={ap!r}"
                )

    def _path_in_container(self, local_path: str) -> str:
        self._validate_paths_under_cwd(local_path)
        rel = os.path.relpath(
            os.path.abspath(local_path),
            start=os.path.abspath(os.getcwd()),
        )
        return rel.replace(os.sep, "/")

    def _compute_resources(self) -> tuple[int, int]:
        num_cpu = os.cpu_count() - 4 if os.cpu_count() and os.cpu_count() > 4 else 1
        memory_mb = int(
            self.memory_fraction * psutil.virtual_memory().total / 1024 / 1024
        )
        return num_cpu, memory_mb

    def _docker_base_cmd(self) -> list[str]:
        cmd = ["docker", "run", "--rm", "-v", f"{os.getcwd()}:/app", "-w", "/app"]
        if torch.cuda.is_available():
            cmd.extend(["--gpus", "all"])
        cmd.append(self.docker_image)
        return cmd

    @staticmethod
    def _seq_hash(sequence: str) -> str:
        return hashlib.md5(sequence.encode()).hexdigest()[:12]

    @staticmethod
    def _write_fasta(sequences: dict[str, str], output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            for seq_id, seq in sequences.items():
                f.write(f">{seq_id}\n{seq}\n")
        return output_path

    def _ensure_docker_image(self) -> None:
        """Check Docker is available and pull the image if needed."""
        try:
            subprocess.run(
                ["docker", "version"],
                capture_output=True, text=True, check=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "Docker is not installed or not found in PATH. "
                "Please install Docker: https://docs.docker.com/get-docker/"
            )
        except subprocess.CalledProcessError:
            raise RuntimeError(
                "Docker daemon is not running. Please start Docker and try again."
            )

        inspect = subprocess.run(
            ["docker", "image", "inspect", self.docker_image],
            capture_output=True, text=True,
        )
        if inspect.returncode == 0:
            return

        logger.info("Docker image '%s' not found locally. Pulling...", self.docker_image)
        try:
            self._run_docker_command(
                ["docker", "pull", self.docker_image],
                check=True, text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to pull Docker image '{self.docker_image}'. "
                f"Please pull it manually: docker pull {self.docker_image}\n"
                f"Error: {e}"
            )
        logger.info("Successfully pulled Docker image '%s'", self.docker_image)

    def create_db(self, fasta_path: str, db_path: str) -> str:
        """Create an MMseqs2 database from a FASTA file.

        Returns the database path prefix.
        """
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        if os.path.exists(f"{db_path}.dbtype"):
            logger.info("MMseqs2 database already exists at %s", db_path)
            return db_path

        self._ensure_docker_image()
        logger.info("Creating MMseqs2 database from %s...", fasta_path)
        self._validate_paths_under_cwd(fasta_path, db_path)

        self._run_docker_command(
            self._docker_base_cmd() + [
                "createdb",
                self._path_in_container(fasta_path),
                self._path_in_container(db_path),
            ],
            check=True, capture_output=True, text=True,
        )
        logger.info("Database created at %s", db_path)
        return db_path

    def create_index(self, db_path: str, tmp_dir: str = None) -> None:
        """Create an index for an MMseqs2 database to speed up searches."""
        if tmp_dir is None:
            tmp_dir = os.path.join(os.path.dirname(db_path), "tmp_index")
        os.makedirs(tmp_dir, exist_ok=True)

        self._ensure_docker_image()
        logger.info("Creating index for %s...", db_path)
        self._validate_paths_under_cwd(db_path, tmp_dir)

        self._run_docker_command(
            self._docker_base_cmd() + [
                "createindex",
                self._path_in_container(db_path),
                self._path_in_container(tmp_dir),
            ],
            check=True, capture_output=True, text=True,
        )
        logger.info("Index created")

    def search(
        self,
        sequence: str,
        output_dir: str,
        seq_id: Optional[str] = None,
    ) -> str:
        """Search for homologues of a sequence and produce an .a3m MSA file.

        Returns path to the output .a3m file.
        """
        if seq_id is None:
            seq_id = self._seq_hash(sequence)

        seq_output_dir = os.path.join(output_dir, seq_id)
        a3m_output = os.path.join(seq_output_dir, f"{seq_id}.a3m")

        if os.path.exists(a3m_output):
            logger.info("MSA already exists for %s: %s", seq_id, a3m_output)
            return a3m_output

        self._ensure_docker_image()
        os.makedirs(seq_output_dir, exist_ok=True)
        num_cpu, memory_mb = self._compute_resources()

        # Write query FASTA
        query_fasta = os.path.join(seq_output_dir, "query.fasta")
        self._write_fasta({seq_id: sequence}, query_fasta)

        # Paths
        query_db = os.path.join(seq_output_dir, "queryDB")
        result_db = os.path.join(seq_output_dir, "resultDB")
        tmp_dir = os.path.join(seq_output_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        self._validate_paths_under_cwd(
            query_fasta, query_db, self.target_db, seq_output_dir, result_db, tmp_dir
        )

        docker_base = self._docker_base_cmd()

        # Step 1: Create query database
        self._run_docker_command(
            docker_base + [
                "createdb",
                self._path_in_container(query_fasta),
                self._path_in_container(query_db),
            ],
            check=True, capture_output=True, text=True,
        )

        # Step 2: Search
        search_cmd = docker_base + [
            "search",
            self._path_in_container(query_db),
            self._path_in_container(self.target_db),
            self._path_in_container(result_db),
            self._path_in_container(tmp_dir),
            "-s", str(self.sensitivity),
            "--max-seqs", str(self.max_seqs),
            "--min-seq-id", str(self.min_seq_id),
            "-c", str(self.coverage),
            "--threads", str(num_cpu),
            "--split-memory-limit", f"{memory_mb}M",
        ]
        if torch.cuda.is_available():
            search_cmd.extend(["--gpu", "1"])

        logger.info("Searching homologues for %s...", seq_id)
        self._run_docker_command(search_cmd, check=True, capture_output=True, text=True)

        # Step 3: Convert results to MSA (a3m format)
        self._run_docker_command(
            docker_base + [
                "result2msa",
                self._path_in_container(query_db),
                self._path_in_container(self.target_db),
                self._path_in_container(result_db),
                self._path_in_container(a3m_output),
                "--msa-format-mode", "6",
            ],
            check=True, capture_output=True, text=True,
        )

        # Clean up intermediate files
        for pattern in ["queryDB*", "resultDB*"]:
            for f in Path(seq_output_dir).glob(pattern):
                f.unlink(missing_ok=True)
        tmp_path = Path(tmp_dir)
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)

        logger.info("MSA saved to %s", a3m_output)
        return a3m_output

    def batch_search(
        self,
        sequences: list[str],
        output_dir: str,
        seq_ids: Optional[list[str]] = None,
        continue_on_error: bool = True,
    ) -> dict[str, str]:
        """Search for homologues for multiple sequences.

        Returns dictionary mapping sequences to their .a3m file paths.
        """
        if seq_ids is None:
            seq_ids = [self._seq_hash(seq) for seq in sequences]

        os.makedirs(output_dir, exist_ok=True)
        results = {}

        to_process = []
        for seq, sid in zip(sequences, seq_ids):
            a3m_path = os.path.join(output_dir, sid, f"{sid}.a3m")
            if os.path.exists(a3m_path):
                results[seq] = a3m_path
            else:
                to_process.append((seq, sid))

        if not to_process:
            logger.info("All MSAs already exist, skipping search.")
            return results

        logger.info(
            "Searching homologues for %d sequences (%d already cached)",
            len(to_process), len(results),
        )

        for seq, sid in tqdm(to_process, desc="Searching homologues"):
            try:
                a3m_path = self.search(
                    sequence=seq,
                    output_dir=output_dir,
                    seq_id=sid,
                )
                results[seq] = a3m_path
            except Exception as e:
                logger.error("Failed to search homologues for %s: %s", sid, e)
                if not continue_on_error:
                    raise

        logger.info("Successfully processed %d/%d sequences", len(results), len(sequences))
        return results


def create_mmseqs_db(
    fasta_path: str,
    db_path: str,
    docker_image: str = DOCKER_IMAGE,
) -> str:
    """Create an MMseqs2 database from a FASTA file."""
    return HomologueSearcher(
        target_db="", docker_image=docker_image,
    ).create_db(fasta_path, db_path)


def create_mmseqs_index(
    db_path: str,
    tmp_dir: str = None,
    docker_image: str = DOCKER_IMAGE,
) -> None:
    """Create an index for an MMseqs2 database."""
    HomologueSearcher(
        target_db="", docker_image=docker_image,
    ).create_index(db_path, tmp_dir)


def search_homologues(
    sequence: str,
    target_db: str,
    output_dir: str,
    seq_id: Optional[str] = None,
    sensitivity: float = 7.5,
    max_seqs: int = 1000,
    min_seq_id: float = 0.0,
    coverage: float = 0.8,
    docker_image: str = DOCKER_IMAGE,
    memory_fraction: float = 0.5,
) -> str:
    """Search for homologues of a sequence and produce an .a3m MSA file."""
    return HomologueSearcher(
        target_db=target_db,
        docker_image=docker_image,
        sensitivity=sensitivity,
        max_seqs=max_seqs,
        min_seq_id=min_seq_id,
        coverage=coverage,
        memory_fraction=memory_fraction,
    ).search(sequence, output_dir, seq_id)


def batch_search_homologues(
    sequences: list[str],
    target_db: str,
    output_dir: str,
    seq_ids: Optional[list[str]] = None,
    sensitivity: float = 7.5,
    max_seqs: int = 1000,
    min_seq_id: float = 0.0,
    coverage: float = 0.8,
    docker_image: str = DOCKER_IMAGE,
    memory_fraction: float = 0.5,
    continue_on_error: bool = True,
) -> dict[str, str]:
    """Search for homologues for multiple sequences."""
    return HomologueSearcher(
        target_db=target_db,
        docker_image=docker_image,
        sensitivity=sensitivity,
        max_seqs=max_seqs,
        min_seq_id=min_seq_id,
        coverage=coverage,
        memory_fraction=memory_fraction,
    ).batch_search(sequences, output_dir, seq_ids, continue_on_error)


class ColabFoldSearcher:
    """ColabFold API-based homologue search. Mirrors HomologueSearcher's interface."""

    def __init__(
        self,
        host_url: str = COLABFOLD_HOST,
        user_agent: str = "",
        mode: str = "env",
        timeout: float = 30.0,
        max_retries: int = 10,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        inter_request_delay: tuple[float, float] = (1.0, 3.0),
        max_wait_time: int = 600,
    ):
        """
        Args:
            mode: "env" (default) | "all" | "env-nofilter" | "nofilter".
                "env" uses UniRef + Environmental DBs with filtering.
            user_agent: Set to your email for the User-Agent header.
            inter_request_delay: (min, max) seconds between requests in batch_search.
            max_wait_time: Seconds before raising TimeoutError on a single job.
        """
        self.host_url = host_url.rstrip("/")
        self.mode = mode
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.inter_request_delay = inter_request_delay
        self.max_wait_time = max_wait_time

        self._session = requests.Session()
        if user_agent:
            self._session.headers["User-Agent"] = user_agent

    @staticmethod
    def _seq_hash(sequence: str) -> str:
        return hashlib.md5(sequence.encode()).hexdigest()[:12]

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter."""
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        jitter = random.uniform(0, delay * 0.5)
        return delay + jitter

    def _request_with_retries(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> requests.Response:
        """HTTP request with automatic retry on 429, 5xx, timeouts, etc."""
        for attempt in range(self.max_retries):
            try:
                if method.upper() == "GET":
                    res = self._session.get(url, timeout=self.timeout, **kwargs)
                else:
                    res = self._session.post(url, timeout=self.timeout, **kwargs)

                if res.status_code == 429:
                    retry_after = float(
                        res.headers.get("Retry-After", self._backoff_delay(attempt))
                    )
                    logger.warning(
                        "Rate limited (HTTP 429). Waiting %.1fs...", retry_after
                    )
                    time.sleep(retry_after)
                    continue

                if res.status_code >= 500:
                    delay = self._backoff_delay(attempt)
                    logger.warning(
                        "Server error %d. Retrying in %.1fs...",
                        res.status_code, delay,
                    )
                    time.sleep(delay)
                    continue

                return res

            except requests.exceptions.Timeout:
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "Timeout (attempt %d/%d). Retrying in %.1fs...",
                    attempt + 1, self.max_retries, delay,
                )
                time.sleep(delay)
            except requests.exceptions.ConnectionError:
                delay = self._backoff_delay(attempt)
                logger.warning("Connection error. Retrying in %.1fs...", delay)
                time.sleep(delay)
            except Exception as e:
                delay = self._backoff_delay(attempt)
                logger.warning("Request error: %s. Retrying in %.1fs...", e, delay)
                time.sleep(delay)

        raise RuntimeError(
            f"Request to {url} failed after {self.max_retries} attempts"
        )

    def _submit(self, sequence: str, mode: Optional[str] = None) -> dict:
        """Submit a sequence for MSA search. Retries on RATELIMIT/UNKNOWN with backoff."""
        mode = mode or self.mode
        query = f">101\n{sequence}\n"

        for attempt in range(self.max_retries):
            res = self._request_with_retries(
                "POST",
                f"{self.host_url}/ticket/msa",
                data={"q": query, "mode": mode},
            )

            try:
                data = res.json()
            except ValueError:
                logger.error("Non-JSON response from server: %s", res.text[:200])
                delay = self._backoff_delay(attempt)
                time.sleep(delay)
                continue

            status = data.get("status")

            if status == "RATELIMIT":
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "API rate limit (attempt %d/%d). Waiting %.1fs...",
                    attempt + 1, self.max_retries, delay,
                )
                time.sleep(delay)
                continue

            if status == "UNKNOWN":
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "Unknown status from API. Retrying in %.1fs...", delay
                )
                time.sleep(delay)
                continue

            return data

        raise RuntimeError(
            f"Failed to submit sequence after {self.max_retries} attempts "
            "(persistent rate-limiting)"
        )

    def _poll(self, ticket_id: str) -> dict:
        """Poll a submitted job until completion or timeout. Interval adapts from 1s to 5s."""
        total_wait = 0
        poll_interval = 1.0          # start fast
        poll_interval_cap = 5.0      # never wait longer than this per poll

        while True:
            res = self._request_with_retries(
                "GET", f"{self.host_url}/ticket/{ticket_id}"
            )

            try:
                data = res.json()
            except ValueError:
                logger.error("Non-JSON polling response: %s", res.text[:200])
                data = {"status": "ERROR"}

            status = data.get("status")

            if status in ("COMPLETE", "ERROR"):
                return data

            if status not in ("RUNNING", "PENDING", "UNKNOWN"):
                return data

            wait = min(poll_interval + random.uniform(0, 0.5), poll_interval_cap)
            logger.info(
                "Job %s status: %s. Waiting %.1fs...", ticket_id, status, wait
            )
            time.sleep(wait)
            total_wait += wait
            poll_interval = min(poll_interval + 1.0, poll_interval_cap)

            if total_wait > self.max_wait_time:
                raise TimeoutError(
                    f"Job {ticket_id} did not complete within "
                    f"{self.max_wait_time}s"
                )

    def _download(self, ticket_id: str, output_path: str) -> None:
        """Download the result archive for a completed job."""
        res = self._request_with_retries(
            "GET", f"{self.host_url}/result/download/{ticket_id}"
        )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(res.content)

    def _extract_a3m(
        self, tar_path: str, output_dir: str, seq_id: str
    ) -> str:
        """Extract .a3m files from the result archive and combine them."""
        with tarfile.open(tar_path) as tar:
            tar.extractall(output_dir)

        uniref_a3m = os.path.join(output_dir, "uniref.a3m")
        env_a3m = os.path.join(
            output_dir, "bfd.mgnify30.metaeuk30.smag30.a3m"
        )

        a3m_files = []
        if os.path.exists(uniref_a3m):
            a3m_files.append(uniref_a3m)
        if "env" in self.mode and os.path.exists(env_a3m):
            a3m_files.append(env_a3m)

        combined_path = os.path.join(output_dir, f"{seq_id}.a3m")

        if len(a3m_files) == 1:
            os.rename(a3m_files[0], combined_path)
        elif len(a3m_files) > 1:
            with open(combined_path, "w") as out_f:
                for a3m_file in a3m_files:
                    with open(a3m_file) as in_f:
                        out_f.write(in_f.read())
        else:
            raise RuntimeError("No .a3m files found in downloaded archive")

        # Clean up intermediate files
        if os.path.exists(tar_path):
            os.remove(tar_path)
        for f in a3m_files:
            if os.path.exists(f) and f != combined_path:
                os.remove(f)

        return combined_path

    def search(
        self,
        sequence: str,
        output_dir: str,
        seq_id: Optional[str] = None,
    ) -> str:
        """Search for homologues of a sequence and produce an .a3m MSA file.

        Returns path to the output .a3m file.
        """
        if seq_id is None:
            seq_id = self._seq_hash(sequence)

        seq_output_dir = os.path.join(output_dir, seq_id)
        a3m_output = os.path.join(seq_output_dir, f"{seq_id}.a3m")

        if os.path.exists(a3m_output):
            logger.info("MSA already exists for %s: %s", seq_id, a3m_output)
            return a3m_output

        os.makedirs(seq_output_dir, exist_ok=True)

        # Submit
        logger.info(
            "Submitting %s for ColabFold MSA search (mode=%s)...",
            seq_id, self.mode,
        )
        result = self._submit(sequence)

        if result.get("status") == "ERROR":
            raise RuntimeError(f"ColabFold API error for {seq_id}")
        if result.get("status") == "MAINTENANCE":
            raise RuntimeError("ColabFold API is under maintenance")

        ticket_id = result["id"]
        logger.info("Job submitted for %s. Ticket: %s", seq_id, ticket_id)

        # Poll for completion
        result = self._poll(ticket_id)

        if result.get("status") != "COMPLETE":
            raise RuntimeError(
                f"Job failed for {seq_id}: {result.get('status')}"
            )

        # Download and extract
        logger.info("Downloading results for %s...", seq_id)
        tar_path = os.path.join(seq_output_dir, f"{seq_id}.tar.gz")
        self._download(ticket_id, tar_path)

        a3m_path = self._extract_a3m(tar_path, seq_output_dir, seq_id)
        logger.info("MSA saved to %s", a3m_path)
        return a3m_path

    def batch_search(
        self,
        sequences: list[str],
        output_dir: str,
        seq_ids: Optional[list[str]] = None,
        continue_on_error: bool = True,
    ) -> dict[str, str]:
        """Search for homologues for multiple sequences. Inserts inter_request_delay between API calls.

        Returns dictionary mapping sequences to their .a3m file paths.
        """
        if seq_ids is None:
            seq_ids = [self._seq_hash(seq) for seq in sequences]

        os.makedirs(output_dir, exist_ok=True)
        results: dict[str, str] = {}

        to_process: list[tuple[str, str]] = []
        for seq, sid in zip(sequences, seq_ids):
            a3m_path = os.path.join(output_dir, sid, f"{sid}.a3m")
            if os.path.exists(a3m_path):
                results[seq] = a3m_path
            else:
                to_process.append((seq, sid))

        if not to_process:
            logger.info("All MSAs already cached. Skipping ColabFold search.")
            return results

        logger.info(
            "ColabFold batch search: %d to process (%d cached)",
            len(to_process), len(results),
        )

        for i, (seq, sid) in enumerate(
            tqdm(to_process, desc="ColabFold search")
        ):
            try:
                a3m_path = self.search(
                    sequence=seq, output_dir=output_dir, seq_id=sid,
                )
                results[seq] = a3m_path
            except Exception as e:
                logger.error("Failed for %s: %s", sid, e)
                if not continue_on_error:
                    raise

            if i < len(to_process) - 1:
                delay = random.uniform(*self.inter_request_delay)
                logger.debug("Waiting %.1fs before next request...", delay)
                time.sleep(delay)

        logger.info(
            "Processed %d/%d sequences via ColabFold",
            len(results), len(sequences),
        )
        return results

def colabfold_search_homologues(
    sequence: str,
    output_dir: str,
    seq_id: Optional[str] = None,
    host_url: str = COLABFOLD_HOST,
    user_agent: str = "",
    mode: str = "env",
    timeout: float = 30.0,
    max_retries: int = 10,
    inter_request_delay: tuple[float, float] = (1.0, 3.0),
) -> str:
    """Convenience wrapper around ColabFoldSearcher.search."""
    return ColabFoldSearcher(
        host_url=host_url,
        user_agent=user_agent,
        mode=mode,
        timeout=timeout,
        max_retries=max_retries,
        inter_request_delay=inter_request_delay,
    ).search(sequence, output_dir, seq_id)


def colabfold_batch_search_homologues(
    sequences: list[str],
    output_dir: str,
    seq_ids: Optional[list[str]] = None,
    host_url: str = COLABFOLD_HOST,
    user_agent: str = "",
    mode: str = "env",
    timeout: float = 30.0,
    max_retries: int = 10,
    inter_request_delay: tuple[float, float] = (1.0, 3.0),
    continue_on_error: bool = True,
) -> dict[str, str]:
    """Convenience wrapper around ColabFoldSearcher.batch_search."""
    return ColabFoldSearcher(
        host_url=host_url,
        user_agent=user_agent,
        mode=mode,
        timeout=timeout,
        max_retries=max_retries,
        inter_request_delay=inter_request_delay,
    ).batch_search(sequences, output_dir, seq_ids, continue_on_error)