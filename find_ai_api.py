import urllib.request
import re

urls = [
    'https://static.cdnpub.info/v5/static/client/iq.1c1e08d9c67b13c465de.js',
    'https://static.cdnpub.info/v5/static/client/main.6871ec051370f4d61a55.js'
]

for u in urls:
    print('Downloading', u)
    data = urllib.request.urlopen(u).read().decode('utf-8', errors='ignore')
    print('Length:', len(data))
    matches = re.findall(r'["\'`][^"\'`]*ai-integration[^"\'`]*["\'`]', data, re.IGNORECASE)
    print('ai-integration matches:', set(matches))
    matches2 = re.findall(r'["\'`]/api/[^"\'`]*token[^"\'`]*["\'`]', data, re.IGNORECASE)
    print('/api/ token matches:', list(set(matches2))[:10])
