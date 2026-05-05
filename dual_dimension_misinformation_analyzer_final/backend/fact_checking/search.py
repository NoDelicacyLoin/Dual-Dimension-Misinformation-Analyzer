import os
from dataclasses import dataclass

import requests


@dataclass
class SearchResult:
    evidence: list[dict]
    error_type: str = ""
    error_message: str = ""


HARD_BLOCK_URL_MARKERS = ["sitemap", "rss", ".xml"]

SOFT_DOCUMENT_URL_MARKERS = [
    "/wp-content/uploads/",
    "/uploads/",
    ".pdf",
    "download",
    "fileshare",
]

def looks_like_low_quality_result(evidence_url: str, evidence_text: str) -> bool:
    lower_url = evidence_url.lower()
    word_count = len(evidence_text.split())

    if any(marker in lower_url for marker in HARD_BLOCK_URL_MARKERS):
        return True

    if any(marker in lower_url for marker in SOFT_DOCUMENT_URL_MARKERS):
        return word_count < 120

    return False


def looks_like_page_shell(search_query: str, evidence_url: str, evidence_text: str) -> bool:
    """
    Check whether the evidence text looks like page shell content
    rather than a usable article snippet.
    """
    lower_url = evidence_url.lower()
    lower_text = evidence_text.lower()
    lower_query = search_query.lower()

    navigation_marker_count = sum(
        marker in lower_text
        for marker in ("jump to content", "table of contents", "skip to main content")
    )
    strong_listing_marker_count = sum(
        marker in lower_text
        for marker in ("fact-checks on", "older posts", "newer posts", "archive", "archives", "all fact-checks")
    )
    weak_listing_marker_count = (
        (1 if "page" in lower_text and "page" not in lower_query else 0)
        + (1 if "category" in lower_text and "category" not in lower_query else 0)
        + (1 if "tagged" in lower_text and "tagged" not in lower_query else 0)
    )

    word_count = len(evidence_text.split())
    stated_on_count = lower_text.count("stated on")
    url_looks_like_list_page = any(
        marker in lower_url
        for marker in ("/list", "/archive", "/archives", "/category/", "/tag/", "?page=")
    )

    looks_like_listing_page = (
        strong_listing_marker_count >= 2
        or stated_on_count >= 2
        or (
            url_looks_like_list_page
            and (
                strong_listing_marker_count >= 1
                or weak_listing_marker_count >= 1
                or stated_on_count >= 1
            )
        )
    )

    if navigation_marker_count >= 3:
        return True

    if looks_like_listing_page:
        return True

    return word_count < 10 and navigation_marker_count >= 1


def deduplicate_evidence_items(evidence_items: list[dict]) -> tuple[list[dict], int]:
    """
    Remove obvious duplicate results.
    Dedup rules:
    1. same URL
    2. same normalized content
    """
    deduplicated_items = []
    seen_urls = set()
    seen_contents = set()
    removed_count = 0

    for evidence_item in evidence_items:
        item_url = evidence_item.get("url", "").strip()
        item_content = evidence_item.get("content", "").strip()
        normalized_content = " ".join(item_content.lower().split())

        if item_url and item_url in seen_urls:
            removed_count += 1
            continue

        if normalized_content and normalized_content in seen_contents:
            removed_count += 1
            continue

        if item_url:
            seen_urls.add(item_url)

        if normalized_content:
            seen_contents.add(normalized_content)

        deduplicated_items.append(evidence_item)

    return deduplicated_items, removed_count


def fetch_search_evidence(search_query: str, max_results: int = 10) -> SearchResult:
    """
    Fetch one Tavily result set for later evidence filtering.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return SearchResult(
            evidence=[],
            error_type="missing_api_key",
            error_message="Tavily API key not found or configured incorrectly.",
        )

    search_url = "https://api.tavily.com/search"

    request_payload = {
        "query": search_query,
        "search_depth": "basic",
        "include_answer": False,
        "max_results": max_results,
    }
    request_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    evidence_list = []

    try:
        api_response = requests.post(
            search_url,
            json=request_payload,
            headers=request_headers,
            timeout=12
        )
        api_response.raise_for_status()

        response_json = api_response.json()
        retrieved_results = response_json.get("results", [])

        for article in retrieved_results:
            article_content = article.get("content", "").strip()
            article_url = article.get("url", "Unknown URL")

            if not article_content:
                continue

            if looks_like_low_quality_result(article_url, article_content):
                continue

            looks_like_shell = looks_like_page_shell(search_query, article_url, article_content)

            if looks_like_shell:
                continue

            evidence_list.append({
                "url": article_url,
                "content": article_content,
            })

        evidence_list, _ = deduplicate_evidence_items(evidence_list)

        if not evidence_list:
            return SearchResult(
                evidence=[],
                error_type="no_results",
                error_message="No reliable evidence found for this claim.",
            )

        return SearchResult(evidence=evidence_list)

    except Exception as error:
        print(f"[tavily] search failed for query {search_query!r}: {type(error).__name__}: {error}")
        return SearchResult(
            evidence=[],
            error_type="api_error",
            error_message="No evidence found due to an unexpected search error.",
        )
