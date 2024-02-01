import json
import os
from pathlib import Path

import numpy as np
def is_json_file(filename):
    return any(filename.endswith(extension) for extension in ["json"])

def parse_json_file(filepath, metric):
    filepath = Path(filepath)
    name = filepath.name.split(".")[0]
    with filepath.open("r") as f:
        try:
            data = json.load(f)
        except json.decoder.JSONDecodeError as err:
            print(f'Error reading file "{filepath}"')
            raise err

    if "results" in data:
        results = data["results"]
    else:
        results = data

    if metric not in results:
        raise ValueError(
            f'Error: metric "{metric}" not available.'
            f' Available metrics: {", ".join(results.keys())}'
        )

    try:
        if metric == "ms-ssim":
            # Convert to db
            values = np.array(results[metric])
            results[metric] = -10 * np.log10(1 - values)

        return {
            # "name": data.get("name", name),
            "xs": results["bpp"], #            "xs": results["bitrate"],
            "ys": results[metric],
        }
    except KeyError:
        raise ValueError(f'Invalid file "{filepath}"')
seq_result_path = '/media/sugon/新加卷/wff/SCC/Experiment/DIC-Bench_v1.0/traditional-video-vtmscc'
seq_results = [os.path.join(seq_result_path, x) for x in os.listdir(seq_result_path) if is_json_file(x)]
for seq_result in sorted(seq_results):
    path_name,file_name = os.path.split(seq_result)
    os.path.splitext(seq_result)
    # reslut = parse_json_file(seq_result,'ms-ssim-rgb')
    # print(file_name,'bpp,msssim;',reslut['xs'],';',reslut['ys'])
    reslut = parse_json_file(seq_result,'psnr-rgb')
    print(file_name,'bpp,psnr-rgb;',reslut['xs'],';',reslut['ys'])