import requests
from PIL import Image
from io import BytesIO
import numpy as np

HEADERS = {'User-Agent': 'ApartmentRoomClassifier/1.0 (educational; https://huggingface.co/Scampolonii)'}
API = 'https://commons.wikimedia.org/w/api.php'

def is_good(img):
    arr = np.array(img.resize((50, 50))).astype(float)
    return arr.std(axis=2).mean() > 10

def find_in_category(category, fname, skip=0):
    found = 0
    cont = {}
    while True:
        params = {
            'action': 'query', 'list': 'categorymembers',
            'cmtitle': f'Category:{category}', 'cmtype': 'file',
            'cmlimit': 50, 'format': 'json', **cont
        }
        resp = requests.get(API, params=params, headers=HEADERS, timeout=10).json()
        members = resp.get('query', {}).get('categorymembers', [])
        for m in members:
            title = m['title']
            if not title.lower().endswith(('.jpg', '.jpeg')):
                continue
            try:
                info = requests.get(API, params={
                    'action': 'query', 'titles': title, 'prop': 'imageinfo',
                    'iiprop': 'url', 'format': 'json',
                }, headers=HEADERS, timeout=10).json()
                for page in info.get('query', {}).get('pages', {}).values():
                    url = page.get('imageinfo', [{}])[0].get('url', '')
                    if not url:
                        continue
                    r = requests.get(url, headers=HEADERS, timeout=15)
                    img = Image.open(BytesIO(r.content)).convert('RGB')
                    w, h = img.size
                    if w < 800 or h < 600:
                        continue
                    if not is_good(img):
                        continue
                    if found < skip:
                        found += 1
                        print(f'  skip [{found}]: {title}')
                        continue
                    img.save(f'examples/{fname}.jpg', 'JPEG', quality=90)
                    print(f'OK {fname}: {w}x{h} — {title}')
                    return True
            except Exception:
                continue
        if 'continue' not in resp:
            break
        cont = {'cmcontinue': resp['continue']['cmcontinue']}
    return False

print('kitchen...')
find_in_category('Residential kitchens', 'kitchen', skip=0)

print('living_room...')
find_in_category('Living rooms', 'living_room', skip=2)
