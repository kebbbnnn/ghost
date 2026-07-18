"""Print client-bootstrap content from chatgpt.com HTML"""
import re
import json
from curl_cffi.requests import Session

s = Session(impersonate="chrome131")
resp = s.get("https://chatgpt.com/", headers={
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
})
html = resp.text

# Extract content inside <script type="application/json" id="client-bootstrap">...</script>
match = re.search(r'<script[^>]*id="client-bootstrap"[^>]*>(.*?)</script>', html, re.DOTALL)
if match:
    bootstrap_text = match.group(1).strip()
    print("Found client-bootstrap script tag!")
    try:
        data = json.loads(bootstrap_text)
        # Print first level keys
        print(f"Keys: {list(data.keys())}")
        
        # Check if there is anything related to geolocation, ip, or country
        for k, v in data.items():
            if isinstance(v, dict):
                # Search recursively or print keys
                subkeys = list(v.keys())
                print(f"  {k} keys: {subkeys}")
                # Print any matched geo/ip values
                for sk in subkeys:
                    if any(term in sk.lower() for term in ["ip", "geo", "country", "location", "lat", "lon"]):
                        print(f"    Match: {sk} = {v[sk]}")
            else:
                if any(term in k.lower() for term in ["ip", "geo", "country", "location", "lat", "lon"]):
                    print(f"  Match: {k} = {v}")
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        print(bootstrap_text[:1000])
else:
    print("Could not find client-bootstrap script tag!")
