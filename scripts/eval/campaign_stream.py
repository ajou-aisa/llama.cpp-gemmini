from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import threading
from pathlib import Path

from eval_common import require, write_json


def run_compressed(argv: list[str], output: Path, raw: Path, timeout: int) -> None:
    read_fd, write_fd = os.pipe()
    expired = threading.Event()
    with os.fdopen(read_fd, "rb") as source, os.fdopen(write_fd, "wb") as target, \
            raw.open("xb") as compressed, gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0, compresslevel=1) as sink, \
            (output / "process.log").open("x", encoding="utf-8") as log:
        command = [*argv, "--scale-output-fd", str(target.fileno())]
        with subprocess.Popen(command, cwd=output, stdout=log, stderr=subprocess.STDOUT,
                              pass_fds=(target.fileno(),)) as process:
            def stop() -> None:
                expired.set()
                process.kill()

            timer = threading.Timer(timeout, stop)
            target.close()
            timer.start()
            try:
                shutil.copyfileobj(source, sink, 1024 * 1024)
                code = process.wait()
            finally:
                timer.cancel()
                if process.poll() is None:
                    process.kill()
                    process.wait()
    write_json(output / "command.json", {"argv": list(command), "cwd": str(output), "exit_code": code,
               "timeout": expired.is_set(), "sink_transport": "inherited_descriptor_to_gzip"})
    require(code == 0 and not expired.is_set(), "compressed native collection failed; see process.log")
