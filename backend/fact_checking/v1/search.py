#search.py
import requests
import os
import re


def analyze_page_shell_features(evidence_url: str, evidence_text: str) -> tuple[bool, list[str]]:
    """
    Check whether the evidence text looks like page shell content
    such as navigation links, archive pages, list pages, or page chrome.
    Return:
    - True / False
    - a list of reasons for the decision
    """
    lower_url = evidence_url.lower()
    lower_text = evidence_text.lower()
    shell_reasons = []

    shell_markers = [
        "jump to content",
        "table of contents",
        "references",
        "external links",
        "add topic",
        "skip to main content"
    ]
    list_page_markers = [
        "fact-checks on",
        "page ",
        "older posts",
        "newer posts",
        "archive",
        "archives",
        "category:",
        "tagged",
        "showing results",
        "browse all",
        "all stories",
        "all fact-checks"
    ]

    marker_count = sum(marker in lower_text for marker in shell_markers)
    list_marker_count = sum(marker in lower_text for marker in list_page_markers)
    link_count = evidence_text.count("](")
    word_count = len(evidence_text.split())
    stated_on_count = lower_text.count("stated on")
    title_heading_count = len(re.findall(r"^#", evidence_text, flags=re.MULTILINE))
    url_looks_like_list_page = any(
        marker in lower_url
        for marker in ["/list", "/archive", "/archives", "/category/", "/tag/", "?page="]
    )

    if marker_count >= 3:
        shell_reasons.append("many_navigation_markers")

    if list_marker_count >= 2:
        shell_reasons.append("many_list_page_markers")

    if stated_on_count >= 2:
        shell_reasons.append("repeated_factcheck_listing")

    if title_heading_count >= 3 and list_marker_count >= 1:
        shell_reasons.append("many_headings_like_listing")

    if url_looks_like_list_page and (list_marker_count >= 1 or stated_on_count >= 1):
        shell_reasons.append("list_like_url")

    if link_count >= 5:
        shell_reasons.append("too_many_links")

    # Keep very short fragments, but do not throw them away too aggressively.
    # Some useful fact-check snippets are short and direct.
    if word_count < 10 and (marker_count >= 1 or link_count >= 3):
        shell_reasons.append("too_short")

    looks_like_shell = len(shell_reasons) > 0
    return looks_like_shell, shell_reasons


def deduplicate_evidence_items(evidence_items: list[dict]) -> tuple[list[dict], int]:
    """
    Remove obvious duplicate results.
    Dedup rules:
    1. same URL
    2. same content prefix
    """
    deduplicated_items = []
    seen_urls = set()
    seen_prefixes = set()
    removed_count = 0

    for evidence_item in evidence_items:
        item_url = evidence_item.get("url", "").strip()
        item_content = evidence_item.get("content", "").strip()
        content_prefix = item_content[:200].strip().lower()

        if item_url and item_url in seen_urls:
            removed_count += 1
            continue

        if content_prefix and content_prefix in seen_prefixes:
            removed_count += 1
            continue

        if item_url:
            seen_urls.add(item_url)

        if content_prefix:
            seen_prefixes.add(content_prefix)

        deduplicated_items.append(evidence_item)

    return deduplicated_items, removed_count


def fetch_oversampled_evidence(optimized_query: str, max_results: int = 8) -> list[dict]:
    """
    Takes an ALREADY OPTIMIZED query and fetches an oversampled list of results 
    from Tavily API. It purposefully fetches more results than needed so the NLI 
    model can filter out the noise later.
    """
    api_key = os.environ.get("Tavily_API_KEY")
    if not api_key:
        print("[Search Error] Tavily API key not found.")
        return [{"url": "Error", "content": "Tavily API key not found or configured incorrectly."}]

    search_url = "https://api.tavily.com/search"

    # 屏蔽极度不靠谱的网站 (Domain Blocklist)
    unreliable_domains = [
        "theonion.com", "reddit.com", "quora.com", 
        "tiktok.com", "twitter.com", "x.com"
    ]

    request_payload = {
        "api_key": api_key,
        "query": optimized_query,  # 这里接收的是已经被 Gemini 提纯过的句子
        "search_depth": "basic",
        "include_answer": False,
        "max_results": max_results,  # 默认抓取 8 条，为后续 NLI 过滤提供充足弹药
        "exclude_domains": unreliable_domains
    }

    evidence_list = []
    filtered_shell_log = []

    retrieved_results_count = 0
    deduplicated_count = 0

    try:
        #print(f"\n[Search API] Fetching top {max_results} results for: '{optimized_query}'...")
        api_response = requests.post(search_url, json=request_payload, timeout=12)
        api_response.raise_for_status()

        response_json = api_response.json()
        retrieved_results = response_json.get("results", [])
        # CHANGED: record how many results Tavily returned before filtering
        retrieved_results_count = len(retrieved_results)

        for article in retrieved_results:

            article_content = article.get("content", "").strip()
            article_url = article.get("url", "Unknown URL")

            if not article_content:
                continue

            looks_like_shell, shell_reasons = analyze_page_shell_features(article_url, article_content)

            if looks_like_shell:
                filtered_shell_log.append({
                    "url": article_url,
                    "content_preview": article_content[:300],
                    "shell_reasons": shell_reasons
                })
                continue

            evidence_list.append({
                "url": article_url,
                "content": article_content
            })

        evidence_list, deduplicated_count = deduplicate_evidence_items(evidence_list)

        # 如果被黑名单全拦截了，或者确实什么都没搜到
        if not evidence_list:
            return [{"url": "Error", "content": "No reliable evidence found for this claim."}]

        print(f"[Search] Retrieved {retrieved_results_count} results | Filtered shell: {len(filtered_shell_log)} | Deduplicated: {deduplicated_count}")

        return evidence_list

    except Exception as e:
        print(f"[Search API Error] {e}")
        return [{"url": "Error", "content": "No evidence found due to an unexpected search error."}]


# --- 本地测试区块 ---
if __name__ == "__main__":
    # 假设这是 Gemini 提纯后的句子
    mock_optimized_query = "Water is not liquid at certain temperatures."
    test_evidence = fetch_oversampled_evidence(mock_optimized_query)

    print("\nOversampled Evidence returned by the function:")
    for i, evidence_item in enumerate(test_evidence):
        print(f"[{i+1}] URL: {evidence_item.get('url')}")
        print(f"    Content snippet: {evidence_item.get('content')[:100]}...\n")
