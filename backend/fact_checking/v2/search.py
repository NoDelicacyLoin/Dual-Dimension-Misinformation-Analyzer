import os
from dataclasses import dataclass

import requests


@dataclass
class SearchResult:
    evidence: list[dict]
    error_type: str = ""
    error_message: str = ""


HARD_BLOCK_URL_MARKERS = [
    "facebook.com",
    "/groups/",
    "/posts/",
    "sitemap",
    "rss",
    ".xml",
]

SOFT_DOCUMENT_URL_MARKERS = [
    "/wp-content/uploads/",
    "/uploads/",
    ".pdf",
    "download",
    "fileshare",
]

LOW_SUBSTANCE_TEXT_MARKERS = [
    "request for proposals",
    "notice inviting bids",
    "bid opening",
    "department of public works",
    "table of contents bidding",
]

EXCLUDED_DOMAINS = [
    "theonion.com",
    "reddit.com",
    "quora.com",
    "tiktok.com",
    "twitter.com",
    "x.com",
    "facebook.com",
]


def get_source_quality(evidence_url: str) -> tuple[str, float]:
    lower_url = evidence_url.lower()

    if any(marker in lower_url for marker in ("politifact.com", "factcheck.org", "snopes.com")):
        return "fact_check", 0.95
    if any(marker in lower_url for marker in (".gov", ".edu", "cbo.gov", "senate.gov", "house.gov")):
        return "official", 0.90
    if any(marker in lower_url for marker in ("reuters.com", "apnews.com", "bbc.com", "nytimes.com", "pbs.org")):
        return "major_news", 0.82
    if "wikipedia.org" in lower_url:
        return "reference", 0.72
    return "general_web", 0.65


def looks_like_low_quality_result(evidence_url: str, evidence_text: str) -> bool:
    lower_url = evidence_url.lower()
    lower_text = evidence_text.lower()
    word_count = len(evidence_text.split())
    low_substance_text_match = any(marker in lower_text for marker in LOW_SUBSTANCE_TEXT_MARKERS)

    if any(marker in lower_url for marker in HARD_BLOCK_URL_MARKERS):
        return True

    if any(marker in lower_url for marker in SOFT_DOCUMENT_URL_MARKERS):
        return word_count < 120 or low_substance_text_match

    if low_substance_text_match:
        return word_count < 80

    return False


def analyze_page_shell_features(search_query: str, evidence_url: str, evidence_text: str) -> bool:
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
        for marker in ["/list", "/archive", "/archives", "/category/", "/tag/", "?page="]
    )

    looks_like_listing_page = (
        strong_listing_marker_count >= 2
        or stated_on_count >= 2
        or (url_looks_like_list_page and (strong_listing_marker_count >= 1 or weak_listing_marker_count >= 1 or stated_on_count >= 1))
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


def fetch_oversampled_evidence(search_query: str, max_results: int = 8) -> SearchResult:
    """
    Fetch an oversampled list of results from Tavily so later filtering has
    enough material to work with.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        print("[Search Error] Tavily API key not found.")
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
        "exclude_domains": EXCLUDED_DOMAINS,
    }
    request_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    evidence_list = []
    retrieved_results_count = 0
    deduplicated_count = 0
    filtered_shell_count = 0

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
        retrieved_results_count = len(retrieved_results)

        for article in retrieved_results:
            article_content = article.get("content", "").strip()
            article_url = article.get("url", "Unknown URL")

            if not article_content:
                continue

            if looks_like_low_quality_result(article_url, article_content):
                filtered_shell_count += 1
                continue

            looks_like_shell = analyze_page_shell_features(search_query, article_url, article_content)

            if looks_like_shell:
                filtered_shell_count += 1
                continue

            source_quality, source_quality_score = get_source_quality(article_url)

            evidence_list.append({
                "url": article_url,
                "content": article_content,
                "source_quality": source_quality,
                "source_quality_score": source_quality_score,
            })

        evidence_list, deduplicated_count = deduplicate_evidence_items(evidence_list)

        if not evidence_list:
            return SearchResult(
                evidence=[],
                error_type="no_results",
                error_message="No reliable evidence found for this claim.",
            )

        print(
            f"[Search] Retrieved {retrieved_results_count} results | "
            f"Filtered shell: {filtered_shell_count} | Deduplicated: {deduplicated_count}"
        )
        return SearchResult(evidence=evidence_list)

    except Exception as error:
        print(f"[Search API Error] {error}")
        return SearchResult(
            evidence=[],
            error_type="api_error",
            error_message="No evidence found due to an unexpected search error.",
        )
