"""Check if cfIp variables are present in chatgpt.com HTML"""
import re
from curl_cffi.requests import Session

s = Session(impersonate="chrome131")
resp = s.get("https://chatgpt.com/", headers={
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
})
html = resp.text

print("--- Searching for cfIp occurrences ---")
matches = re.findall(r'[^"\']*(?:cfIp|cfConnecting)[^"\']*', html, re.IGNORECASE)
for m in matches[:10]:
    print(m.strip()[:150])

print("\n--- Searching for any geolocation keywords (Latitude, Longitude) ---")
geo_matches = re.findall(r'[^"\']*(?:latitude|longitude|country|city)[^"\']*', html, re.IGNORECASE)
for m in geo_matches[:10]:
    print(m.strip()[:150])
