# Shot Labels

Place your label files here. One `.txt` file per video.

## Format

```
# Video: [exact filename]
# Date labeled: [date]
# Notes: [optional context]

MM:SS shot_type
MM:SS shot_type
...
```

## Valid shot types
- `serve`
- `forehand`
- `backhand`
- `forehand_volley`
- `backhand_volley`
- `overhead`

## Example

File: `IMG_1234_labels.txt`
```
# Video: IMG_1234.mp4
# Date labeled: 2025-02-25
# Notes: Practice session, clear back view

0:05 serve
0:12 forehand
0:15 backhand
0:23 forehand_volley
0:31 overhead
0:45 serve
1:02 backhand
1:15 forehand
```

## After labeling

Run: `python scripts/process_labels.py labels/IMG_1234_labels.txt`

This will:
1. Extract clips around each timestamp
2. Generate pose data
3. Add to training set
4. Update progress report
