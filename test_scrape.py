"""Find the build number from chatgpt.com HTML"""
import re
from curl_cffi.requests import Session

s = Session(impersonate="chrome146")
resp = s.get("https://chatgpt.com/", headers={
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
})
html = resp.text

# Search for all 7-digit numbers near "build" context
for m in re.finditer(r'["\']?(\d{7,8})["\']?', html):
    start = max(0, m.start() - 60)
    end = min(len(html), m.end() + 60)
    context = html[start:end].replace('\n', ' ')
    if any(kw in context.lower() for kw in ['build', 'version', 'number', 'client']):
        print(f"Number: {m.group(1)} | Context: ...{context}...")

# Also search for buildNumber or similar patterns in script content
print("\n--- Searching script tags ---")
scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL)
for i, script in enumerate(scripts):
    if 'build' in script.lower() and len(script) < 5000:
        # Find build-related lines
        for line in script.split('\n'):
            if 'build' in line.lower():
                print(f"Script {i}: {line.strip()[:200]}")

# Try fetching a known endpoint that might reveal the build number
print("\n--- Checking meta tags ---")
meta_tags = re.findall(r'<meta[^>]+>', html)
for tag in meta_tags:
    if 'build' in tag.lower() or 'version' in tag.lower():
        print(f"Meta: {tag}")

# Look at data attributes
data_attrs = re.findall(r'data-[a-z-]+="[^"]*build[^"]*"', html, re.IGNORECASE)
for attr in data_attrs:
    print(f"Data attr: {attr}")

# Check the Next.js chunk manifest for build hash
print("\n--- Next.js paths ---")
next_paths = re.findall(r'/_next/static/([a-zA-Z0-9_-]+)/', html)
if next_paths:
    print(f"Build hashes: {set(next_paths)}")

# Search for the specific format we need
print("\n--- Looking for numeric build ID ---")
for pattern in [
    r'"buildNumber":\s*"?(\d+)"?',
    r'"build_number":\s*"?(\d+)"?',
    r'buildId["\s:]+([a-zA-Z0-9_-]+)',
    r'"b"\s*:\s*"(\d{7,})"',
    r'window\.__BUILD_NUMBER__\s*=\s*"?(\d+)"?',
]:
    for m in re.finditer(pattern, html):
        print(f"  {pattern}: {m.group(1)}")
