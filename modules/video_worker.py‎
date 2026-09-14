"""Background task queue for image-to-video jobs."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

import args_manager
import modules.config
from modules.generation_lock import generation_lock
from modules.private_logger import log_video
from modules.util import generate_temp_filename
from modules.video_models import (
    get_model,
    model_is_ready,
    preflight,
    resolve_hardware_profile,
    runtime_is_ready,
    runtime_python,
)


@dataclass
class VideoTask:
    image: np.ndarray | str
    prompt: str
    model: str = "wan"
    hardware_profile: str = "Auto"
    resolution: str = "Auto"
    duration: float = 5.0
    seed: int = 0
    negative_prompt: str = ""
    guidance_scale: float = 5.0
    yields: list = field(default_factory=list)
    processing: bool = False
    cancel_requested: bool = False
    process: subprocess.Popen | None = None

    def cancel(self) -> None:
        self.cancel_requested = True
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()


video_tasks: list[VideoTask] = []
_queue_lock = threading.Lock()
_worker_started = False


def submit(task: VideoTask) -> None:
    with _queue_lock:
        video_tasks.append(task)


def _save_input_image(image: np.ndarray | str, path: str) -> None:
    if isinstance(image, str):
        source = Image.open(image).convert("RGB")
    else:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        source = Image.fromarray(array).convert("RGB")
    source.save(path, format="PNG")


def _emit(task: VideoTask, event: str, payload) -> None:
    task.yields.append([event, payload])


def _unload_image_models() -> None:
    try:
        import ldm_patched.modules.model_management as model_management

        model_management.unload_all_models()
        model_management.soft_empty_cache(force=True)
    except Exception as exc:
        print(f"[Video] Could not fully clear image models: {exc}")


def _reader_thread(stream, messages: queue.Queue) -> None:
    try:
        for line in iter(stream.readline, ""):
            messages.put(line.rstrip())
    finally:
        messages.put(None)


def _handle_runtime_message(task: VideoTask, line: str, result: dict) -> None:
    if not line:
        return
    try:
        message = json.loads(line)
    except json.JSONDecodeError:
        print(f"[Video runtime] {line}")
        return

    event = message.get("event")
    if event == "progress":
        _emit(
            task,
            "progress",
            (
                int(message.get("percent", 0)),
                str(message.get("message", "Generating video …")),
            ),
        )
    elif event == "complete":
        result.update(message)
    elif event == "error":
        result["error"] = message.get("message", "Video runtime failed.")
    elif event == "cancelled":
        result["cancelled"] = True


def _validate_task(task: VideoTask) -> tuple[str, str]:
    if task.image is None:
        raise ValueError("Choose an input image before generating a video.")
    if not runtime_is_ready():
        raise RuntimeError("Video runtime is not installed. Click “Download / Set up” first.")
    if not model_is_ready(task.model):
        raise RuntimeError(f"{get_model(task.model).label} is not downloaded.")
    profile = resolve_hardware_profile(task.hardware_profile)
    allowed, message = preflight(task.model, profile)
    if not allowed:
        raise RuntimeError(message)
    return profile, message


def _create_request(task: VideoTask, profile: str) -> tuple[Path, Path, str]:
    work_dir = Path(modules.config.temp_path) / "video"
    work_dir.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{time.time_ns()}"
    input_path = work_dir / f"{token}-input.png"
    request_path = work_dir / f"{token}-request.json"
    _save_input_image(task.image, str(input_path))
    output_root = modules.config.temp_path if args_manager.args.disable_image_log else modules.config.path_outputs
    _date, output_path, _name = generate_temp_filename(folder=output_root, extension="mp4")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    request = {
        "model": task.model,
        "model_path": str(get_model(task.model).path),
        "image_path": str(input_path),
        "output_path": output_path,
        "prompt": task.prompt.strip(),
        "negative_prompt": task.negative_prompt.strip(),
        "profile": profile,
        "resolution": task.resolution,
        "duration": float(task.duration),
        "seed": int(task.seed),
        "guidance_scale": float(task.guidance_scale),
    }
    request_path.write_text(json.dumps(request), encoding="utf-8")
    return input_path, request_path, output_path


def _read_runtime(task: VideoTask, messages: queue.Queue) -> dict:
    result: dict = {}
    while True:
        if task.cancel_requested and task.process and task.process.poll() is None:
            task.process.terminate()
        try:
            line = messages.get(timeout=0.1)
        except queue.Empty:
            continue
        if line is None:
            return result
        _handle_runtime_message(task, line, result)


def _execute_runtime(task: VideoTask, request_path: Path) -> dict:
    command = [
        str(runtime_python()),
        "-m",
        "modules.video_runtime",
        "--request",
        str(request_path),
    ]
    task.process = subprocess.Popen(
        command,
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if task.process.stdout is None:
        raise RuntimeError("Could not read output from the video runtime.")
    messages: queue.Queue = queue.Queue()
    threading.Thread(
        target=_reader_thread,
        args=(task.process.stdout, messages),
        daemon=True,
    ).start()
    result = _read_runtime(task, messages)
    return_code = task.process.wait()
    if task.cancel_requested or result.get("cancelled"):
        raise InterruptedError("Video generation cancelled.")
    if result.get("error"):
        raise RuntimeError(result["error"])
    if return_code != 0:
        raise RuntimeError(f"Video runtime stopped with exit code {return_code}.")
    if not result.get("path") or not Path(result["path"]).is_file():
        raise RuntimeError("Video runtime completed without producing an MP4 file.")
    return result


def _save_result(task: VideoTask, profile: str, started: float, result: dict) -> None:
    metadata = {
        "Model": get_model(task.model).label,
        "Prompt": task.prompt,
        "Negative prompt": task.negative_prompt if task.model == "wan" else "",
        "Seed": task.seed,
        "Hardware profile": profile,
        "Frames": result.get("frames"),
        "FPS": result.get("fps"),
        "Dimensions": f"{result.get('width')}×{result.get('height')}",
        "Audio": bool(result.get("audio")),
        "Generation time seconds": round(time.perf_counter() - started, 2),
    }
    log_video(result["path"], metadata)
    _emit(task, "finish", result["path"])


def _remove_files(*paths) -> None:
    for path in paths:
        if path:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass


def process_task(task: VideoTask) -> None:
    task.processing = True
    started = time.perf_counter()
    input_path = request_path = output_path = None
    completed = False
    try:
        if task.cancel_requested:
            raise InterruptedError("Video generation cancelled.")
        profile, message = _validate_task(task)
        _emit(task, "progress", (1, message))
        _unload_image_models()
        input_path, request_path, output_path = _create_request(task, profile)
        result = _execute_runtime(task, request_path)
        _save_result(task, profile, started, result)
        completed = True
    except InterruptedError as exc:
        _emit(task, "cancelled", str(exc))
    except Exception as exc:
        traceback.print_exc()
        _emit(task, "error", str(exc))
    finally:
        task.processing = False
        task.process = None
        _remove_files(input_path, request_path)
        if not completed:
            _remove_files(output_path)


def worker() -> None:
    while True:
        task = None
        with _queue_lock:
            if video_tasks:
                task = video_tasks.pop(0)
        if task is None:
            time.sleep(0.05)
            continue
        with generation_lock:
            process_task(task)


def start_worker() -> None:
    global _worker_started
    if _worker_started:
        return
    thread = threading.Thread(target=worker, daemon=True, name="fooocus-video-worker")
    thread.start()
    _worker_started = True


start_worker()