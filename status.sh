#!/bin/bash
# Quick status check for tennis pipeline

echo "=== Pipeline Status ==="
curl -s http://localhost:8080/stats | python3 -c "import sys,json; s=json.load(sys.stdin); print(f'Pending: {s[\"pending\"]} | Processing: {s[\"processing\"]} | Completed: {s[\"completed\"]} | Failed: {s[\"failed\"]}')"

echo ""
echo "=== Jobs ==="
curl -s http://localhost:8080/jobs | python3 -c "
import sys,json
for j in json.load(sys.stdin)['jobs']:
    fn = j['filename'][:20]
    st = j['status']
    yt = (j.get('youtube_url') or '-')[:35]
    print(f'{fn:20} | {st:10} | {yt}')
"
