"""Reusable Gradio card linking to a visual guide."""

from html import escape
from urllib.parse import urlparse

import gradio as gr


def tutorial(guide_url: str) -> gr.HTML:
    """Create a full-width clickable card that opens a guide in a new tab.

    Call this function while building a Gradio Blocks interface, for example::

        tutorial("https://github.com/owner/repo/blob/dev/guide.md")

    Args:
        guide_url: Absolute HTTP(S) URL of the visual guide.

    Returns:
        A Gradio HTML component containing the guide link card.

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

    card_html = f"""
    <a
        href="{safe_url}"
        target="_blank"
        rel="noopener noreferrer"
        aria-label="Open the visual guide on GitHub"
        style="
            display: flex;
            align-items: center;
            gap: 18px;
            width: 100%;
            box-sizing: border-box;
            margin: 14px 0 18px;
            padding: 20px 22px;
            border: 1px solid rgba(126, 110, 255, 0.48);
            border-radius: 14px;
            background: linear-gradient(110deg, #1c2941 0%, #222746 100%);
            color: #eef1fa;
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
                flex: 0 0 60px;
                width: 60px;
                height: 60px;
                border-radius: 12px;
                background: rgba(125, 107, 255, 0.12);
            "
        >
            <svg
                width="34"
                height="34"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
            >
                <path
                    d="M12 6.5C9.5 4.8 6.2 4.5 3 5.5V19
                       C6.2 18 9.5 18.3 12 20
                       M12 6.5C14.5 4.8 17.8 4.5 21 5.5V19
                       C17.8 18 14.5 18.3 12 20
                       M12 6.5V20"
                    stroke="#9b8cff"
                    stroke-width="1.7"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                />
            </svg>
        </span>

        <span style="flex: 1; min-width: 0;">
            <span
                style="
                    display: block;
                    margin-bottom: 5px;
                    font-size: 20px;
                    font-weight: 650;
                    line-height: 1.25;
                "
            >
                Visual guide
            </span>
            <span
                style="
                    display: block;
                    color: #aeb9d2;
                    font-size: 14px;
                    line-height: 1.45;
                "
            >
                Screenshots, prompts, and step-by-step examples
            </span>
        </span>

        <span
            style="
                flex: 0 0 auto;
                padding: 12px 18px;
                border-radius: 10px;
                background: #5749df;
                color: #ffffff;
                font-size: 14px;
                font-weight: 600;
                white-space: nowrap;
            "
        >
            Open guide &#8599;
        </span>
    </a>
    """

    return gr.HTML(value=card_html)
