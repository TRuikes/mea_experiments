from kilosort.io import save_probe
from pathlib import Path
import numpy as np

print('loaded ')

def load_probe_to_kilosort_format(probe_file, save=True):

    probe_file = Path(probe_file)

    # Execute probe file safely into namespace
    namespace = {}

    with open(probe_file, "r") as f:
        exec(f.read(), namespace)

    if "channel_groups" not in namespace:
        raise ValueError("Probe file does not contain 'channel_groups'")

    channel_groups = namespace["channel_groups"]
    group_id = 1

    chanMap = []
    xc = []
    yc = []
    kcoords = []

    channels = np.arange(0, 256)
    geometry = channel_groups[group_id]["geometry"]

    for ch in channels:

        if ch not in geometry:
            print(ch)
            continue

        x, y = geometry[ch]

        chanMap.append(ch)
        xc.append(float(x))
        yc.append(float(y))
        kcoords.append(int(group_id))

    # Convert to numpy arrays
    chanMap = np.array(chanMap, dtype=np.int32)
    xc = np.array(xc, dtype=np.float32)
    yc = np.array(yc, dtype=np.float32)
    kcoords = np.array(kcoords, dtype=np.int32)

    probe = {
        "chanMap": chanMap,
        "xc": xc,
        "yc": yc,
        "kcoords": kcoords,
        "n_chan": len(chanMap),
    }

    # Save Kilosort-style probe file
    if save:

        output_file = probe_file.with_name(
            probe_file.stem + "_kilosort.json"
        )

        save_probe(probe, output_file)

        print(f"Saved Kilosort probe to:\n{output_file}")

    return probe

load_probe_to_kilosort_format(r"E:\bu_hudrogel\2026-05-12 rat LE 1355 A\raw\256_100_30_mea.prb")