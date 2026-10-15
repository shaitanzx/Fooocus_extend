"""Reusable compact Gradio link to a visual guide on GitHub."""

from html import escape
import os
from pathlib import Path, PurePosixPath
from urllib.parse import quote
import warnings
import pygit2
import gradio as gr

REPOSITORY_URL = "https://github.com/shaitanzx/Fooocus_extend"
EXTENSIONS_DIRECTORY = "extentions"  # This spelling matches the repository path.
DEFAULT_REF = "main"


def _detect_source_ref() -> str:
    """Detect the copied checkout's branch once when this module is imported."""
    configured_ref = os.getenv("FOOOCUS_EXTEND_REF", "").strip()
    if configured_ref:
        return configured_ref

    
    try:
        start_path = str(Path(__file__).resolve().parent)
        git_dir = pygit2.discover_repository(start_path)
        if git_dir:
            repository = pygit2.Repository(git_dir)
            if repository.head_is_detached:
                target = repository.head.target
                if target is not None:
                    return target.hex[:12]
            else:
                branch = repository.head.shorthand
                if branch:
                    return branch
    except (pygit2.GitError, OSError, AttributeError):
        pass

    return DEFAULT_REF


# Compute this once at import time. The Caption call becomes a simple path append.
SOURCE_REF = _detect_source_ref()
EXTENSIONS_BASE_URL = (
    f"{REPOSITORY_URL}/blob/{quote(SOURCE_REF, safe='/')}/{EXTENSIONS_DIRECTORY}/"
)


def tutorial(guide_path: str) -> gr.HTML:
    """Create a compact, full-width visual-guide link styled as a button.

    Pass a path relative to the repository's ``extentions/`` folder, as Caption does::

        tutorial("caption/readme.md")

    Args:
        guide_path: Relative POSIX path inside ``extentions/``.

    Returns:
        A Gradio HTML component containing the clickable guide bar.

    Raises:
        TypeError: If ``guide_path`` is not a string.
        ValueError: If it is empty, absolute, or attempts to leave ``extentions/``.
    """
    if not isinstance(guide_path, str):
        raise TypeError("guide_path must be a string")

    guide_path = guide_path.strip()
    if not guide_path or "\\" in guide_path:
        raise ValueError("guide_path must be a non-empty relative POSIX path")

    relative_path = PurePosixPath(guide_path)
    if relative_path.is_absolute() or any(part in ("", ".", "..") for part in relative_path.parts):
        raise ValueError("guide_path must stay inside the extentions/ directory")

    guide_url = EXTENSIONS_BASE_URL + quote(relative_path.as_posix(), safe="/")

    # Escape the URL before inserting it into an HTML attribute.
    safe_url = escape(guide_url, quote=True)

    button_html = f"""
    <a
        href="{safe_url}"
        target="_blank"
        rel="noopener noreferrer"
        aria-label="Open the guide on GitHub"
        style="
            display: flex;
            align-items: center;
            gap: 12px;
            width: 100%;
            min-height: 36px;
            box-sizing: border-box;
            margin: 8px 0 10px;
            padding: 4px 16px;
            border: 1px solid rgba(126, 145, 255, 0.82);
            border-radius: 14px;
            background: linear-gradient(100deg, #1b2943 0%, #202b49 100%);
            color: #dfe7fb;
            text-decoration: none;
            cursor: pointer;
        "
    >
        <span
            aria-hidden="true"
            style="
                display: flex;
                align-items: center;
                justify-content: center;
                flex: 0 0 26px;
                width: 26px;
                height: 26px;
            "
        >
            <svg
                width="26"
                height="26"
                viewBox="0 0 40 40"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
            >
                <path
                    d="M20 11.5C15.5 8.5 9.5 8.3 4 10.5V31
                       C9.5 28.8 15.5 29 20 32
                       M20 11.5C24.5 8.5 30.5 8.3 36 10.5V31
                       C30.5 28.8 24.5 29 20 32
                       M20 11.5V32"
                    stroke="#89A0FF"
                    stroke-width="2.2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                />
                <path
                    d="M31 3.5V10.5M27.5 7H34.5"
                    stroke="#A795FF"
                    stroke-width="2.2"
                    stroke-linecap="round"
                />
            </svg>
        </span>

        <span
            style="
                flex: 1 1 auto;
                min-width: 0;
                color: #e1e8f8;
                font-size: 14px;
                font-weight: 550;
                line-height: 1.25;
            "
        >
            Need help? Open the guide
        </span>

        <span
            style="
                flex: 0 0 auto;
                margin-left: auto;
                color: #91a6ff;
                font-size: 13px;
                font-weight: 550;
                white-space: nowrap;
            "
        >
            View on GitHub &#8599;
        </span>
    </a>
    """

    return gr.HTML(value=button_html)
