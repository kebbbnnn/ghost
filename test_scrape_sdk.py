"""Scrape assets for sentinel references"""
import re
import urllib.parse
from curl_cffi.requests import Session

s = Session(impersonate="chrome131")
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://chatgpt.com/",
}

# 1. Fetch main page to extract asset URLs
resp = s.get("https://chatgpt.com/", headers=headers)
html = resp.text

assets = re.findall(r'src="(/cdn/assets/[^"]+)"', html)
assets += re.findall(r'href="(/cdn/assets/[^"]+)"', html)
assets = list(set(assets))

print(f"Found {len(assets)} assets to scan.")

for asset in assets:
    if not asset.endswith(".js"):
        continue
    url = urllib.parse.urljoin("https://chatgpt.com", asset)
    print(f"Fetching and scanning: {url}")
    asset_resp = s.get(url, headers=headers)
    text = asset_resp.text
    
    # Search for sentinel or sdk.js references
    if "sentinel" in text.lower() or "sdk.js" in text.lower():
        print(f"*** MATCH FOUND IN {asset} ***")
        # Print a window around the match
        for m in re.finditer(r'sentinel|sdk\.js', text, re.IGNORECASE):
            start = max(0, m.start() - 100)
            end = min(len(text), m.end() + 100)
            print(f"  Context: ...{text[start:end].strip()}...")
