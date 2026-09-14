"""Local model and runtime management for Fooocus image-to-video (Cross-platform with detailed logging)."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import threading
import time
import venv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import modules.config

# ==============================================================================
# 🔧 ЛОГИРОВАНИЕ
# ==============================================================================
def _log(message: str) -> None:
    """Выводит сообщение с меткой времени и префиксом модуля."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [VideoModels] {message}", flush=True)

# ==============================================================================
# 🔧 КРОССПЛАТФОРМЕННАЯ ИНИЦИАЛИЗАЦИЯ ПУТЕЙ
# ==============================================================================
_log("Инициализация модуля video_models...")

# Находим корневую папку Fooocus (на 1 уровень выше папки modules)
BASE_DIR = Path(__file__).resolve().parents[1]
_log(f"Определена базовая директория проекта: {BASE_DIR}")

# Принудительно задаем абсолютные пути
modules.config.path_video_runtime = str(BASE_DIR / "video_runtime")
_log(f"Путь к video_runtime установлен в: {modules.config.path_video_runtime}")

if not getattr(modules.config, 'path_video_models', None):
    modules.config.path_video_models = str(BASE_DIR / "models" / "video_models")
    _log(f"Путь к video_models не был задан, установлен по умолчанию: {modules.config.path_video_models}")
else:
    orig_path = Path(modules.config.path_video_models)
    if not orig_path.is_absolute():
        modules.config.path_video_models = str(BASE_DIR / orig_path)
    _log(f"Итоговый путь к video_models: {modules.config.path_video_models}")


GIB = 1024 ** 3
H3_LICENSE_URL = "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"
MODEL_INDEX = "model_index.json"


@dataclass(frozen=True)
class VideoModel:
    key: str
    label: str
    repo_id: str
    folder_name: str
    download_bytes: int
    required_paths: tuple[str, ...]
    allow_patterns: tuple[str, ...] | None = None
    experimental: bool = False

    @property
    def path(self) -> Path:
        return Path(modules.config.path_video_models) / self.folder_name


VIDEO_MODELS = {
    "wan": VideoModel(
        key="wan",
        label="Wan 2.2 TI2V 5B",
        repo_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        folder_name="wan2.2-ti2v-5b",
        download_bytes=34_000_000_000,
        required_paths=(MODEL_INDEX, "transformer", "text_encoder", "tokenizer", "vae"),
    ),
    "h3": VideoModel(
        key="h3",
        label="MiniMax H3-Base FL2VA (experimental)",
        repo_id="MiniMaxAI/MiniMax-H3",
        folder_name="minimax-h3-fl2va",
        download_bytes=150_000_000_000,
        required_paths=("modular_model_index.json", "transformer", "text_encoder", "tokenizer", "processor", "vae", "audio_vae", "scheduler", "audio_scheduler"),
        allow_patterns=(MODEL_INDEX, "modular_model_index.json", "transformer/**", "text_encoder/**", "tokenizer/**", "processor/**", "vae/**", "audio_vae/**", "scheduler/**", "audio_scheduler/**"),
        experimental=True,
    ),
}


def get_model(model_key: str) -> VideoModel:
    _log(f"Запрос модели по ключу: '{model_key}'")
    try:
        model = VIDEO_MODELS[model_key]
        _log(f"Модель найдена: {model.label}")
        return model
    except KeyError as exc:
        _log(f"ОШИБКА: Неизвестный ключ модели '{model_key}'")
        raise ValueError(f"Unknown video model: {model_key}") from exc


def model_components_present(model_key: str) -> bool:
    model = get_model(model_key)
    _log(f"Проверка наличия компонентов для {model_key} в папке: {model.path}")
    missing = [p for p in model.required_paths if not (model.path / p).exists()]
    if missing:
        _log(f"Отсутствуют компоненты: {missing}")
        return False
    _log("Все требуемые компоненты присутствуют.")
    return True


def model_is_ready(model_key: str) -> bool:
    _log(f"Проверка готовности модели: {model_key}")
    components_ok = model_components_present(model_key)
    marker_path = Path(modules.config.path_video_models) / get_model(model_key).folder_name / ".fooocus-complete"
    marker_ok = marker_path.is_file()
    
    if components_ok and marker_ok:
        _log(f"Модель {model_key} ПОЛНОСТЬЮ ГОТОВА к использованию.")
        return True
    else:
        _log(f"Модель {model_key} НЕ ГОТОВА (компоненты: {components_ok}, маркер: {marker_ok}).")
        return False


def model_status(model_key: str) -> str:
    model = get_model(model_key)
    if model_is_ready(model_key):
        return f"Ready: {model.path}"
    size_gb = model.download_bytes / 1_000_000_000
    return f"Not installed · about {size_gb:.0f} GB · {model.path}"


def runtime_python() -> Path:
    root = Path(modules.config.path_video_runtime)
    if os.name == "nt":
        python_path = root / "Scripts" / "python.exe"
        _log(f"Определена ОС Windows. Путь к Python в venv: {python_path}")
    else:
        python_path = root / "bin" / "python"
        _log(f"Определена ОС Linux/macOS (Colab). Путь к Python в venv: {python_path}")
    return python_path


def _video_requirements() -> Path:
    req_path = Path(__file__).resolve().parents[1] / "requirements_video.txt"
    _log(f"Поиск файла требований: {req_path}")
    return req_path


def _requirements_hash() -> str:
    req_file = _video_requirements()
    if not req_file.exists():
        _log(f"ВНИМАНИЕ: Файл требований {req_file} не найден. Создается заглушка.")
        req_file.parent.mkdir(parents=True, exist_ok=True)
        req_file.write_text("torch\ndiffusers\n")
    
    file_hash = hashlib.sha256(req_file.read_bytes()).hexdigest()
    _log(f"Хеш файла требований: {file_hash[:16]}...")
    return file_hash


def runtime_is_ready() -> bool:
    _log("Проверка готовности видео-окружения (runtime)...")
    python = runtime_python()
    if not python.is_file():
        _log(f"-> Не готово: исполняемый файл Python не найден по пути {python}")
        return False
    
    # 🔧 НОВАЯ ПРОВЕРКА: убеждаемся, что pip тоже существует
    if os.name == "nt":
        pip_path = python.parent / "pip.exe"
    else:
        pip_path = python.parent / "pip"
    
    if not pip_path.is_file():
        _log(f"-> Не готово: pip не найден по пути {pip_path}. venv повреждён, требуется пересоздание.")
        return False
        
    marker = Path(modules.config.path_video_runtime) / ".fooocus-video-runtime"
    if not marker.is_file():
        _log(f"-> Не готово: маркер окружения не найден по пути {marker}")
        return False
        
    try:
        state = json.loads(marker.read_text(encoding="utf-8"))
        current_hash = _requirements_hash()
        saved_hash = state.get("requirements_hash")
        
        if saved_hash == current_hash:
            _log("-> Готово: хеш требований совпадает, окружение актуально.")
            return True
        else:
            _log(f"-> Не готово: хеш требований изменился (был: {saved_hash[:8]}..., стал: {current_hash[:8]}...). Требуется переустановка.")
            return False
    except (OSError, ValueError) as e:
        _log(f"-> Не готово: ошибка чтения маркера ({e}).")
        return False


def setup_runtime(progress: Callable[[str], None] | None = None) -> Path:
    _log("=== НАЧАЛО setup_runtime ===")
    report = progress or (lambda _message: None)
    runtime_dir = Path(modules.config.path_video_runtime)
    requirements = _video_requirements()
    
    _log(f"Создание директорий (если отсутствуют): {runtime_dir.parent} и {runtime_dir}")
    runtime_dir.parent.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    if runtime_is_ready():
        msg = "✅ Video environment is already up to date."
        _log(msg)
        report(msg)
        return runtime_python()

    # 🔧 ИСПРАВЛЕНИЕ: принудительно удаляем битый venv, если он есть
    python_exe = runtime_python()
    if python_exe.exists():
        _log(f"⚠️ Обнаружен неполный venv по пути {python_exe.parent}. Удаляем для пересоздания...")
        try:
            shutil.rmtree(runtime_dir)
            _log("Старый venv удалён.")
        except Exception as e:
            _log(f"Не удалось полностью удалить старый venv: {e}. Пробуем продолжить...")
        runtime_dir.mkdir(parents=True, exist_ok=True)

    # Создаём venv с clear=True для гарантии чистой установки
    msg = "🛠️ Creating isolated video environment (this may take a minute)..."
    _log(msg)
    report(msg)
    
    _log("Запуск venv.EnvBuilder(with_pip=True, clear=True)...")
    try:
        venv.EnvBuilder(with_pip=True, clear=True).create(runtime_dir)
    except Exception as e:
        _log(f"Ошибка при создании venv: {e}. Пробуем альтернативный метод через ensurepip...")
        # Запасной вариант для Colab
        venv.EnvBuilder(with_pip=False, clear=True).create(runtime_dir)
        _log("Запуск ensurepip для установки pip...")
        subprocess.run([str(python_exe), "-m", "ensurepip", "--upgrade"], check=True)
    
    _log("venv успешно создан.")
    
    #  Дополнительная проверка: убеждаемся, что pip теперь есть
    if os.name == "nt":
        pip_exe = python_exe.parent / "pip.exe"
    else:
        pip_exe = python_exe.parent / "pip"
    
    if not pip_exe.exists():
        _log(f"КРИТИЧЕСКАЯ ОШИБКА: pip не был установлен даже после пересоздания venv. Путь: {pip_exe}")
        raise RuntimeError(f"Failed to install pip in venv at {runtime_dir}")
    _log(f"pip найден по пути: {pip_exe}")

    msg = "📦 Installing video runtime packages …"
    _log(msg)
    report(msg)
    
    cmd = [str(python_exe), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(requirements)]
    _log(f"Выполнение команды: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    if process.stdout is None:
        _log("КРИТИЧЕСКАЯ ОШИБКА: Не удалось получить stdout от процесса установки.")
        raise RuntimeError("Could not read output from the video environment installer.")
        
    _log("--- Начало вывода pip install ---")
    for line in process.stdout:
        line = line.strip()
        if line:
            _log(f"  [pip] {line}")
            report(line)
    _log("--- Конец вывода pip install ---")
    
    return_code = process.wait()
    _log(f"Процесс pip install завершен с кодом возврата: {return_code}")
    
    if return_code != 0:
        _log("КРИТИЧЕСКАЯ ОШИБКА: Установка пакетов завершилась неудачно.")
        raise RuntimeError(f"Video environment setup failed with exit code {return_code}.")

    _log("Запись маркера успешной установки окружения...")
    marker = runtime_dir / ".fooocus-video-runtime"
    marker.write_text(
        json.dumps({
            "requirements": str(requirements),
            "requirements_hash": _requirements_hash(),
            "created_at": time.time(),
        }),
        encoding="utf-8",
    )
    
    msg = "✅ Video environment is ready."
    _log(msg)
    report(msg)
    _log("=== ЗАВЕРШЕНИЕ setup_runtime ===")
    return runtime_python()


def _download_model_process(model_key: str, status_file: str) -> None:
    _log(f"[Дочерний процесс] Начата загрузка модели: {model_key}")
    model = get_model(model_key)
    status_path = Path(status_file)
    try:
        from huggingface_hub import snapshot_download

        _log(f"[Дочерний процесс] Запись статуса: Connecting...")
        status_path.write_text("Connecting to Hugging Face …", encoding="utf-8")
        
        _log(f"[Дочерний процесс] Создание директории модели: {model.path}")
        model.path.mkdir(parents=True, exist_ok=True)
        
        _log(f"[Дочерний процесс] Вызов snapshot_download для {model.repo_id}...")
        snapshot_download(
            repo_id=model.repo_id,
            local_dir=str(model.path),
            allow_patterns=list(model.allow_patterns) if model.allow_patterns else None,
            resume_download=True,
        )
        
        _log(f"[Дочерний процесс] Загрузка завершена. Проверка компонентов...")
        if not model_components_present(model_key):
            _log("[Дочерний процесс] ОШИБКА: После загрузки отсутствуют требуемые компоненты.")
            raise RuntimeError("Download completed but required model components are missing.")
            
        _log(f"[Дочерний процесс] Запись маркера .fooocus-complete...")
        (model.path / ".fooocus-complete").write_text(model.repo_id, encoding="utf-8")
        status_path.write_text("complete", encoding="utf-8")
        _log(f"[Дочерний процесс] Загрузка модели {model_key} успешно завершена.")
        
    except Exception as exc:
        _log(f"[Дочерний процесс] КРИТИЧЕСКАЯ ОШИБКА при загрузке: {exc}")
        status_path.write_text(f"error: {exc}", encoding="utf-8")
        raise


class DownloadManager:
    """Owns the current resumable model download so the UI can cancel it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: multiprocessing.Process | None = None
        self._status_file: Path | None = None
        self._model_key: str | None = None

    def start(self, model_key: str) -> None:
        _log(f"DownloadManager.start() вызван для модели: {model_key}")
        with self._lock:
            if self._process is not None and self._process.is_alive():
                _log("ОШИБКА: Попытка запустить загрузку, когда другая уже активна.")
                raise RuntimeError("Another video model download is already running.")
                
            model = get_model(model_key)
            free_bytes = shutil.disk_usage(model.path.parent).free
            current_size = directory_size(model.path)
            remaining = max(0, model.download_bytes - current_size)
            
            _log(f"Проверка диска: Свободно={free_bytes/GIB:.1f} GiB, Уже скачано={current_size/GIB:.1f} GiB, Осталось={remaining/GIB:.1f} GiB")
            
            if free_bytes < remaining + 2 * GIB:
                _log("ОШИБКА: Недостаточно свободного места на диске.")
                raise RuntimeError(
                    f"Not enough free disk space. Need about {remaining / GIB:.1f} GiB "
                    f"plus 2 GiB working space. (Current free: {free_bytes / GIB:.1f} GiB)"
                )
                
            status_dir = Path(modules.config.path_video_runtime)
            status_dir.mkdir(parents=True, exist_ok=True)
            self._status_file = status_dir / "download-status.txt"
            self._status_file.write_text(f"Starting {model.label} download …", encoding="utf-8")
            self._model_key = model_key
            
            _log("Запуск дочернего процесса загрузки через multiprocessing (context='spawn')...")
            ctx = multiprocessing.get_context("spawn")
            self._process = ctx.Process(
                target=_download_model_process,
                args=(model_key, str(self._status_file)),
                daemon=True,
            )
            self._process.start()
            _log(f"Дочерний процесс запущен с PID: {self._process.pid}")

    def cancel(self) -> bool:
        _log("DownloadManager.cancel() вызван.")
        with self._lock:
            if self._process is None or not self._process.is_alive():
                _log("Отмена не требуется: процесс не запущен или уже завершен.")
                return False
                
            _log(f"Отправка сигнала terminate() процессу с PID {self._process.pid}...")
            self._process.terminate()
            self._process.join(timeout=5)
            
            if self._status_file:
                self._status_file.write_text("Cancelled. Run Download again to resume.", encoding="utf-8")
                _log("Статус изменен на 'Cancelled'.")
            return True

    def status(self) -> tuple[bool, str]:
        with self._lock:
            running = self._process is not None and self._process.is_alive()
            if self._status_file and self._status_file.exists():
                message = self._status_file.read_text(encoding="utf-8")
            else:
                message = "Idle"
                
            if running and self._model_key:
                model = get_model(self._model_key)
                downloaded = directory_size(model.path)
                percent = min(99, int(100 * downloaded / model.download_bytes))
                message = (
                    f"{message}\nDownloaded about {downloaded / GIB:.1f} of "
                    f"{model.download_bytes / GIB:.1f} GiB ({percent}%)."
                )
                
            if self._process is not None and not running and self._process.exitcode:
                if not message.startswith("error:") and not message.startswith("Cancelled"):
                    message = f"Download stopped with exit code {self._process.exitcode}."
                    _log(f"ВНИМАНИЕ: Процесс загрузки завершился с кодом {self._process.exitcode}")
                    
            return running, message


DOWNLOAD_MANAGER = DownloadManager()


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    # Логируем только для больших директорий или при первом вызове, чтобы не спамить
    for root, _dirs, files in os.walk(path):
        for file_name in files:
            try:
                total += (Path(root) / file_name).stat().st_size
            except OSError:
                pass
    return total


def hardware_info() -> dict[str, float | str | bool]:
    _log("Сбор информации о железе (hardware_info)...")
    total_ram = 0
    try:
        import psutil
        total_ram = psutil.virtual_memory().total
        _log(f"  System RAM: {total_ram / GIB:.1f} GB")
    except Exception as e:
        _log(f"  Не удалось получить RAM (psutil): {e}")

    vram = 0
    cuda = False
    device_name = "CPU"
    try:
        import torch
        cuda = torch.cuda.is_available()
        _log(f"  CUDA доступен: {cuda}")
        if cuda:
            properties = torch.cuda.get_device_properties(0)
            vram = properties.total_memory
            device_name = properties.name
            _log(f"  GPU: {device_name} ({vram / GIB:.1f} GB VRAM)")
    except Exception as e:
        _log(f"  Не удалось получить информацию о GPU (torch): {e}")

    free_disk = shutil.disk_usage(modules.config.path_video_models).free
    _log(f"  Свободно на диске (video_models): {free_disk / GIB:.1f} GB")
    
    return {
        "cuda": cuda,
        "device_name": device_name,
        "vram_gb": round(vram / GIB, 1),
        "ram_gb": round(total_ram / GIB, 1),
        "free_disk_gb": round(free_disk / GIB, 1),
    }


def resolve_hardware_profile(requested: str = "Auto") -> str:
    if requested != "Auto":
        _log(f"Используется запрошенный профиль железа: {requested}")
        return requested
        
    info = hardware_info()
    vram = float(info["vram_gb"])
    if vram < 10:
        profile = "8 GB"
    elif vram < 14:
        profile = "12 GB"
    else:
        profile = "16 GB+"
        
    _log(f"Автоопределение профиля железа по VRAM ({vram} GB): выбран профиль '{profile}'")
    return profile


def preflight(model_key: str, profile: str = "Auto") -> tuple[bool, str]:
    _log(f"=== ЗАПУСК PREFLIGHT для модели: {model_key}, профиль: {profile} ===")
    model = get_model(model_key)
    info = hardware_info()
    selected = resolve_hardware_profile(profile)
    
    if not info["cuda"]:
        _log("PREFLIGHT ЗАВЕРШЕН С ОШИБКОЙ: Отсутствует CUDA.")
        return False, "A CUDA-capable NVIDIA GPU is required for local video generation."

    if model_key == "h3":
        _log("Выполнение специфичных проверок для H3...")
        if not h3_authorized():
            _log("PREFLIGHT ЗАВЕРШЕН С ОШИБКОЙ: H3 не авторизован.")
            return False, "Confirm your MiniMax H3 authorization before downloading or running H3."
            
        warnings = []
        if float(info["vram_gb"]) < 12:
            warnings.append("H3 is unsupported below 12 GB VRAM")
        if float(info["ram_gb"]) < 75:
            warnings.append("H3 offload normally needs about 75 GB system RAM")
            
        remaining_gib = max(0, model.download_bytes - directory_size(model.path)) / GIB
        if float(info["free_disk_gb"]) < remaining_gib:
            warnings.append(f"Need about {remaining_gib:.0f} GiB more free disk")
            
        if warnings:
            _log(f"PREFLIGHT ЗАВЕРШЕН С ПРЕДУПРЕЖДЕНИЯМИ: {'; '.join(warnings)}")
            return False, "; ".join(warnings) + "."
            
        _log("PREFLIGHT ДЛЯ H3 ПРОШЕЛ УСПЕШНО.")
        return True, "H3 experimental preflight passed."

    _log(f"PREFLIGHT ДЛЯ WAN ПРОШЕЛ УСПЕШНО (Профиль: {selected}).")
    if selected == "8 GB":
        return True, "8 GB mode uses reduced frames/resolution and maximum CPU offload; generation is slow."
    return True, f"Wan profile: {selected}."


def _h3_marker() -> Path:
    return Path(modules.config.path_video_models) / ".minimax-h3-authorization"


def set_h3_authorized(authorized: bool) -> None:
    _log(f"Вызов set_h3_authorized({authorized})")
    marker = _h3_marker()
    if authorized:
        _log("Запись маркера авторизации H3...")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps({"acknowledged": True, "license": H3_LICENSE_URL}),
            encoding="utf-8",
        )
    elif marker.exists():
        _log("Удаление маркера авторизации H3...")
        marker.unlink()


def h3_authorized() -> bool:
    is_auth = _h3_marker().is_file()
    _log(f"Проверка авторизации H3: {'ДА' if is_auth else 'НЕТ'}")
    return is_auth


def model_status_summary() -> str:
    _log("Генерация сводки статуса моделей (model_status_summary)...")
    info = hardware_info()
    lines = [
        f"GPU: {info['device_name']} ({info['vram_gb']} GB VRAM)",
        f"System RAM: {info['ram_gb']} GB · Free disk: {info['free_disk_gb']} GB",
        f"Wan: {model_status('wan')}",
        f"MiniMax H3: {model_status('h3')}",
    ]
    summary = "\n".join(lines)
    _log("Сводка статуса:\n" + summary)
    return summary