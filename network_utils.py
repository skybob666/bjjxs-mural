"""网络请求配置：直连失败时尝试使用Windows已配置的本机代理。"""

from __future__ import annotations

import os
import re

import requests


def use_system_proxy() -> bool:
    """仅在用户明确开启时读取HTTP_PROXY等环境变量。"""
    value = os.getenv("WALL_MURAL_USE_SYSTEM_PROXY", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def windows_proxy_url() -> str | None:
    """读取Windows当前用户代理；无法读取时返回None。"""
    if os.name != "nt":
        return None
    try:
        import winreg

        key_path = r"Software\Microsoft\Windows\CurrentVersion\Internet Settings"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            enabled = int(winreg.QueryValueEx(key, "ProxyEnable")[0] or 0)
            server = str(winreg.QueryValueEx(key, "ProxyServer")[0] or "").strip()
    except (OSError, ValueError):
        return None
    if not enabled or not server:
        return None

    # Windows可以保存“http=host:port;https=host:port”格式。
    entries = {}
    for item in server.split(";"):
        if "=" in item:
            name, value = item.split("=", 1)
            entries[name.strip().lower()] = value.strip()
    candidate = entries.get("https") or entries.get("http") or server
    if not candidate or candidate.lower().startswith("socks"):
        return None
    if not re.match(r"^[a-z][a-z0-9+.-]*://", candidate, flags=re.I):
        candidate = f"http://{candidate}"
    return candidate


def configured_proxy_url() -> str | None:
    """按当前模式读取代理：默认忽略环境代理，优先使用Windows本机代理。"""
    if use_system_proxy():
        return (
            os.getenv("HTTPS_PROXY")
            or os.getenv("https_proxy")
            or os.getenv("HTTP_PROXY")
            or os.getenv("http_proxy")
            or windows_proxy_url()
        )
    return windows_proxy_url()


def create_http_session(prefer_proxy: bool = False) -> requests.Session:
    """创建API会话；默认直连，必要时显式使用已配置代理。"""
    session = requests.Session()
    if prefer_proxy or use_system_proxy():
        proxy = configured_proxy_url()
        if proxy:
            session.trust_env = False
            session.proxies.update({"http": proxy, "https": proxy})
        else:
            session.trust_env = True
    else:
        session.trust_env = False
    return session


def proxy_available() -> bool:
    return configured_proxy_url() is not None
