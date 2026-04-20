"""
Downloads one example image per room category from Wikimedia Commons.
"""
import os
import requests
from PIL import Image
from io import BytesIO

os.makedirs("examples", exist_ok=True)

HEADERS = {"User-Agent": "ApartmentRoomClassifier/1.0 (educational project; https://huggingface.co/Scampolonii)"}

SEARCHES = {
    "bathroom": "modern bathroom interior photograph",
    "bedroom": "apartment bedroom interior photograph",
    "childrens_room": "children bedroom interior photograph",
    "corridor": "apartment hallway corridor interior photograph",
    "dining_room": "modern dining room interior photograph",
    "kitchen": "modern kitchen interior photograph",
    "living_room": "modern living room interior photograph",
    "nursery": "baby nursery room interior photograph",
}

API = "https://commons.wikimedia.org/w/api.php"

def search_and_download(label, query):
    resp = requests.get(API, params={
        "action": "query",
        "list": "search",
        "srsearch": f"{query} filetype:bitmap",
        "srnamespace": 6,
        "srlimit": 5,
        "format": "json",
    }, headers=HEADERS, timeout=10)
    results = resp.json().get("query", {}).get("search", [])
    if not results:
        print(f"  No results for {label}")
        return False

    for result in results:
        title = result["title"]
        info = requests.get(API, params={
            "action": "query",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json",
        }, headers=HEADERS, timeout=10).json()
        pages = info.get("query", {}).get("pages", {})
        for page in pages.values():
            url = page.get("imageinfo", [{}])[0].get("url", "")
            if not url or any(url.endswith(ext) for ext in (".svg", ".tif", ".tiff", ".ogv", ".webm")):
                continue
            try:
                img_resp = requests.get(url, headers=HEADERS, timeout=15)
                img = Image.open(BytesIO(img_resp.content)).convert("RGB")
                path = f"examples/{label}.jpg"
                img.save(path, "JPEG", quality=90)
                print(f"  Saved: {path}  ({img.size})")
                return True
            except Exception as e:
                print(f"  Failed to load image: {e}")
                continue
    print(f"  Could not download any image for {label}")
    return False


for label, query in SEARCHES.items():
    print(f"Downloading: {label}...")
    search_and_download(label, query)

saved = [f for f in os.listdir("examples") if f.endswith(".jpg")]
print(f"\nDone — {len(saved)}/{len(SEARCHES)} images saved to examples/")
