"""Reusable compact Gradio link to a visual guide."""

from html import escape
from urllib.parse import urlparse

import gradio as gr


def tutorial(guide_url: str) -> gr.HTML:
    """Create a compact, full-width visual-guide link styled as a button.

    Call this function while building a Gradio Blocks interface::

        tutorial("https://github.com/owner/repo/blob/dev/guide.md")

    Args:
        guide_url: Absolute HTTP(S) URL of the visual guide.

    Returns:
        A Gradio HTML component containing the clickable guide bar.

    Raises:
        TypeError: If ``guide_url`` is not a string.
        ValueError: If ``guide_url`` is empty or is not an absolute HTTP(S) URL.
    """
    if not isinstance(guide_url, str):
        raise TypeError("guide_url must be a string")

    guide_url = guide_url.strip()
    parsed_url = urlparse(guide_url)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
        raise ValueError("guide_url must be an absolute http:// or https:// URL")

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
