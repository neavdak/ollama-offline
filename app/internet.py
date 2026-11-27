"""Internet search and webpage fetching tools with clean extraction and SSRF guards."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import socket
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("localgravity.internet")

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _is_safe_public_url(url: str) -> tuple[bool, str]:
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Invalid URL"

    if parsed.scheme not in ("http", "https"):
        return False, f"Unsupported scheme: {parsed.scheme}"

    host = parsed.hostname
    if not host:
        return False, "Missing hostname"

    if host.lower() in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return False, "Blocked local/loopback host"

    try:
        infos = socket.getaddrinfo(host, None)
        for info in infos:
            ip_str = info[4][0]
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False, f"Blocked non-public IP: {ip_str}"
    except socket.gaierror:
        # Let httpx surface DNS failure naturally if unresolvable
        pass
    except Exception:
        pass
    return True, ""


def _ddgs_sync(query: str, max_results: int) -> list[dict[str, str]]:
    from ddgs import DDGS

    results: list[dict[str, str]] = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": (item.get("title") or "").strip(),
                    "url": (item.get("href") or item.get("link") or "").strip(),
                    "snippet": (item.get("body") or item.get("snippet") or "").strip(),
                    "source": "ddgs",
                }
            )
    return results


async def _ddg_html_fallback(query: str, max_results: int) -> list[dict[str, str]]:
    headers = {"User-Agent": _USER_AGENT}
    async with httpx.AsyncClient(timeout=12.0, follow_redirects=True, headers=headers) as client:
        resp = await client.post("https://html.duckduckgo.com/html/", data={"q": query})
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    results: list[dict[str, str]] = []
    for block in soup.select(".result"):
        a = block.select_one(".result__a")
        if not a:
            continue
        href = a.get("href", "")
        if "uddg=" in href:
            qs = parse_qs(urlparse(href).query)
            href = unquote(qs.get("uddg", [href])[0])
        snippet_el = block.select_one(".result__snippet")
        results.append(
            {
                "title": a.get_text(" ", strip=True),
                "url": href,
                "snippet": snippet_el.get_text(" ", strip=True) if snippet_el else "",
                "source": "ddg_html",
            }
        )
        if len(results) >= max_results:
            break
    return results


async def _wikipedia_fallback(query: str, max_results: int) -> list[dict[str, str]]:
    headers = {"User-Agent": _USER_AGENT}
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": max_results,
    }
    async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
        resp = await client.get("https://en.wikipedia.org/w/api.php", params=params)
        resp.raise_for_status()
        data = resp.json()

    results: list[dict[str, str]] = []
    for item in data.get("query", {}).get("search", []):
        title = item.get("title", "")
        snippet = BeautifulSoup(item.get("snippet", ""), "html.parser").get_text(" ", strip=True)
        slug = title.replace(" ", "_")
        results.append(
            {
                "title": title,
                "url": f"https://en.wikipedia.org/wiki/{slug}",
                "snippet": snippet,
                "source": "wikipedia",
            }
        )
    return results


async def web_search(query: str, max_results: int = 6) -> dict[str, Any]:
    """Search the live internet using DuckDuckGo with resilient fallbacks."""
    max_results = max(1, min(int(max_results), 12))
    errors: list[str] = []
    logger.info("web_search query=%r max_results=%d", query, max_results)

    try:
        results = await asyncio.wait_for(asyncio.to_thread(_ddgs_sync, query, max_results), timeout=12.0)
        if results:
            logger.info("web_search provider=ddgs count=%d", len(results))
            return {"query": query, "count": len(results), "results": results}
    except Exception as exc:
        errors.append(f"ddgs: {exc}")

    try:
        results = await _ddg_html_fallback(query, max_results)
        if results:
            logger.info("web_search provider=ddg_html count=%d", len(results))
            return {"query": query, "count": len(results), "results": results}
    except Exception as exc:
        errors.append(f"ddg_html: {exc}")

    try:
        results = await _wikipedia_fallback(query, max_results)
        if results:
            logger.info("web_search provider=wikipedia count=%d", len(results))
            return {"query": query, "count": len(results), "results": results}
    except Exception as exc:
        errors.append(f"wikipedia: {exc}")

    logger.warning("web_search returned 0 results for %r errors=%s", query, errors)
    return {"query": query, "count": 0, "results": [], "errors": errors}


def _clean_html_to_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    for tag in soup(["script", "style", "noscript", "svg", "iframe", "nav", "footer", "aside"]):
        tag.decompose()

    root = soup.find("main") or soup.find("article") or soup.body or soup
    lines: list[str] = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "code", "blockquote", "tr"]):
        if el.name == "code" and el.parent and el.parent.name == "pre":
            continue
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            prefix = "#" * int(el.name[1])
            lines.append(f"\n{prefix} {text}")
        elif el.name == "li":
            lines.append(f"- {text}")
        elif el.name == "pre":
            lines.append(f"```\n{el.get_text()}\n```")
        else:
            lines.append(text)

    if not lines:
        raw = root.get_text("\n", strip=True)
        cleaned = re.sub(r"\n{3,}", "\n\n", raw)
        return title, cleaned

    # Deduplicate consecutive identical lines often created by nested elements
    deduped: list[str] = []
    for line in lines:
        if not deduped or deduped[-1] != line:
            deduped.append(line)
    text_out = re.sub(r"\n{3,}", "\n\n", "\n".join(deduped)).strip()
    return title, text_out


async def fetch_url(url: str, max_chars: int = 12000) -> dict[str, Any]:
    """Fetch a public URL and extract clean readable text."""
    safe, reason = _is_safe_public_url(url)
    if not safe:
        return {"url": url, "error": reason}

    max_chars = max(500, min(int(max_chars), 40000))
    headers = {"User-Agent": _USER_AGENT, "Accept": "text/html,application/json,text/plain,*/*"}
    logger.info("fetch_url -> %s", url)
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=headers) as client:
            resp = await client.get(url)
            final_url = str(resp.url)
            content_type = resp.headers.get("content-type", "").lower()
            logger.info("fetch_url status=%s final_url=%s", resp.status_code, final_url)

            if resp.status_code >= 400:
                return {
                    "url": final_url,
                    "status": resp.status_code,
                    "error": f"HTTP {resp.status_code}",
                }

            if "json" in content_type:
                text = resp.text[:max_chars]
                return {
                    "url": final_url,
                    "status": resp.status_code,
                    "title": "JSON Response",
                    "content": text,
                    "truncated": len(resp.text) > max_chars,
                }

            if "html" in content_type or "<html" in resp.text[:500].lower():
                title, extracted = _clean_html_to_text(resp.text)
                truncated = len(extracted) > max_chars
                return {
                    "url": final_url,
                    "status": resp.status_code,
                    "title": title,
                    "content": extracted[:max_chars],
                    "truncated": truncated,
                }

            text = resp.text[:max_chars]
            return {
                "url": final_url,
                "status": resp.status_code,
                "title": final_url,
                "content": text,
                "truncated": len(resp.text) > max_chars,
            }
    except Exception as exc:
        return {"url": url, "error": str(exc)}
