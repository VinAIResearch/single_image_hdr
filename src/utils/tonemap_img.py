import os
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm


class Tonemapper(ABC):
    def __init__(self, out_dir) -> None:
        super().__init__()
        self.out_dir = out_dir

    @abstractmethod
    def saveTonemapped(self, filepath):
        pass

    @staticmethod
    def quantize(ldr):
        return (ldr * 255).clip(min=0.0, max=255.0).round().astype(np.uint8)


class TonemapperReinhard(Tonemapper):
    def __init__(self, out_dir, *args, **kwargs) -> None:
        super().__init__(out_dir)
        self.out_dir = out_dir
        self.processor = cv2.createTonemapReinhard(*args, **kwargs)

    def saveTonemapped(self, filepath):
        hdr = cv2.imread(filepath, -1)
        ldr = self.processor.process(hdr)
        ldr = self.quantize(ldr)

        dirpath, filename = os.path.split(filepath)
        _, scene_name = os.path.split(dirpath)
        _, filename = os.path.split(filepath)
        cv2.imwrite(
            f"{self.out_dir}/{filename[:filename.rfind('.')]}.png",
            ldr,
        )


class TonemapperPhotomatix(Tonemapper):
    def __init__(self, out_dir, preset_path) -> None:
        if out_dir[-1] != "/":
            out_dir += "/"
        super().__init__(out_dir)
        self.preset_path = preset_path

    def saveTonemapped(self, filepath):
        dirpath, filename = os.path.split(filepath)
        # scene_name = os.path.split(dirpath)[1]
        scene_name = filename.split(".")[0]
        os.system(
            f"PhotomatixCL -x {self.preset_path} -bi 16 "
            f"-d {self.out_dir} -o {scene_name} {filepath} >/dev/null 2>&1"
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i")
    parser.add_argument("-o")
    parser.add_argument("--photomatix", action="store_true")
    parser.add_argument("--preset", default="photomatix-presets/photomatix-preset.xmp")

    args = parser.parse_args()

    filepaths = glob(f"{args.i}/**/*.hdr", recursive=True)

    if args.photomatix:
        tonemapper = TonemapperPhotomatix(args.o, args.preset)
    else:
        tonemapper = TonemapperReinhard(args.o, 2.2, 0.0, 0.5, 0.0)

    os.makedirs(args.o, exist_ok=True)

    with ThreadPoolExecutor() as executor:
        with tqdm(total=len(filepaths)) as progress:
            futures = []
            for filepath in filepaths:
                future = executor.submit(tonemapper.saveTonemapped, filepath)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)

            for future in as_completed(futures):
                pass
