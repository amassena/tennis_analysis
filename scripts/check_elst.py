#!/usr/bin/env python3
"""Parse and display MP4 edit list (elst) atoms."""
import struct
import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else "preprocessed/IMG_0994.mp4"
with open(filepath, "rb") as f:
    data = f.read(4096)  # elst is always near the start

pos = 0
while pos < len(data) - 8:
    # Find elst atom
    idx = data.find(b"elst", pos)
    if idx < 0:
        print("No elst atom found")
        break

    atom_size = struct.unpack(">I", data[idx-4:idx])[0]
    print(f"elst atom at byte {idx}, size={atom_size}")

    version = data[idx+4]
    flags = struct.unpack(">I", b'\x00' + data[idx+5:idx+8])[0]
    entry_count = struct.unpack(">I", data[idx+8:idx+12])[0]
    print(f"  version={version}, flags={flags}, entries={entry_count}")

    offset = idx + 12
    for i in range(entry_count):
        if version == 0:
            seg_dur = struct.unpack(">I", data[offset:offset+4])[0]
            media_time = struct.unpack(">i", data[offset+4:offset+8])[0]
            rate_int = struct.unpack(">H", data[offset+8:offset+10])[0]
            rate_frac = struct.unpack(">H", data[offset+10:offset+12])[0]
            print(f"  [{i}] segment_duration={seg_dur} media_time={media_time} "
                  f"media_rate={rate_int}.{rate_frac}")
            offset += 12
        else:
            seg_dur = struct.unpack(">Q", data[offset:offset+8])[0]
            media_time = struct.unpack(">q", data[offset+8:offset+16])[0]
            rate_int = struct.unpack(">H", data[offset+16:offset+18])[0]
            rate_frac = struct.unpack(">H", data[offset+18:offset+20])[0]
            print(f"  [{i}] segment_duration={seg_dur} media_time={media_time} "
                  f"media_rate={rate_int}.{rate_frac}")
            offset += 20

    pos = idx + atom_size
    if pos >= len(data):
        break

# Also check the raw file's edit list
print()
if len(sys.argv) > 2:
    raw_path = sys.argv[2]
    with open(raw_path, "rb") as f:
        raw_data = f.read(8192)
    idx = raw_data.find(b"elst")
    if idx < 0:
        print(f"Raw file: no elst atom")
    else:
        atom_size = struct.unpack(">I", raw_data[idx-4:idx])[0]
        version = raw_data[idx+4]
        entry_count = struct.unpack(">I", raw_data[idx+8:idx+12])[0]
        print(f"Raw file elst: {entry_count} entries, version={version}")
        offset = idx + 12
        for i in range(entry_count):
            if version == 0:
                seg_dur = struct.unpack(">I", raw_data[offset:offset+4])[0]
                media_time = struct.unpack(">i", raw_data[offset+4:offset+8])[0]
                rate_int = struct.unpack(">H", raw_data[offset+8:offset+10])[0]
                rate_frac = struct.unpack(">H", raw_data[offset+10:offset+12])[0]
                print(f"  [{i}] segment_duration={seg_dur} media_time={media_time} "
                      f"media_rate={rate_int}.{rate_frac}")
                offset += 12
            else:
                seg_dur = struct.unpack(">Q", raw_data[offset:offset+8])[0]
                media_time = struct.unpack(">q", raw_data[offset+8:offset+16])[0]
                rate_int = struct.unpack(">H", raw_data[offset+16:offset+18])[0]
                rate_frac = struct.unpack(">H", raw_data[offset+18:offset+20])[0]
                print(f"  [{i}] segment_duration={seg_dur} media_time={media_time} "
                      f"media_rate={rate_int}.{rate_frac}")
                offset += 20
