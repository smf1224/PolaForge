# import os
# from pathlib import Path
# import cv2
# import numpy as np
# from PIL import Image
# import torchvision.transforms as transforms
# from .base_dataset import BaseDataset
# from .image_folder import make_dataset
# from .utils import MaskToTensor
#
# IMG_EXTS = (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg")
#
#
# def safe_imread(path, mode=cv2.IMREAD_UNCHANGED, desc="image"):
#     p = Path(path)
#     img = cv2.imread(str(p), mode)
#     if img is None:
#         raise FileNotFoundError(
#             f"[DATASET] Unable to read {desc}: {p}\n"
#             f"  - exists: {p.exists()}   is_file: {p.is_file()}\n"
#             f"  - hint: check CWD/os.getcwd(), path spelling, case sensitivity, extension"
#         )
#     return img
#
#
# class CrackDataset(BaseDataset):
#     def __init__(self, args):
#         super().__init__(args)
#         self.args = args
#         self.modals = args.modals
#         self.modal_num = len(args.modals)
#         self.modal_img_paths = []
#         for i in range(self.modal_num):
#             modal_dir = os.path.join(args.dataset_path, f"{args.phase}_img_{self.modals[i]}")
#             paths = make_dataset(modal_dir)
#             self.modal_img_paths.append(paths)
#
#         self.lab_dir = Path(args.dataset_path) / f"{args.phase}_lab"
#         self.lab_map = {}
#         if not self.lab_dir.exists():
#             raise FileNotFoundError(f"[DATASET] Label directory not found: {self.lab_dir}")
#         for ext in IMG_EXTS:
#             for p in self.lab_dir.glob(f"*{ext}"):
#                 self.lab_map[p.stem] = str(p)
#         self.img_transforms = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5),
#                                  (0.5, 0.5, 0.5))
#         ])
#         self.lab_transform = MaskToTensor()
#         self.phase = args.phase
#         base_len = len(self.modal_img_paths[0])
#         for i in range(1, self.modal_num):
#             if len(self.modal_img_paths[i]) != base_len:
#                 raise ValueError(
#                     f"[DATASET] Modal length mismatch: {self.modals[0]}={base_len}, "
#                     f"{self.modals[i]}={len(self.modal_img_paths[i])}. "
#                     f"Ensure same count & ordering."
#                 )
#
#     def __len__(self):
#         return len(self.modal_img_paths[0])
#
#     def _find_label_path_by_stem(self, stem: str) -> str:
#         if stem in self.lab_map:
#             return self.lab_map[stem]
#
#         for ext in IMG_EXTS:
#             p = self.lab_dir / f"{stem}{ext}"
#             if p.exists():
#                 self.lab_map[stem] = str(p)
#                 return str(p)
#
#         raise FileNotFoundError(
#             f"[DATASET] Label not found for stem='{stem}' under {self.lab_dir}\n"
#             f"  - tried: {', '.join([stem + e for e in IMG_EXTS])}"
#         )
#
#     def __getitem__(self, index):
#         modal_img_paths = []
#         for i in range(self.modal_num):
#             try:
#                 modal_img_paths.append(self.modal_img_paths[i][index])
#             except IndexError:
#                 raise IndexError(
#                     f"[DATASET] Index out of range for modal '{self.modals[i]}': "
#                     f"index={index}, len={len(self.modal_img_paths[i])}"
#                 )
#
#         stem = Path(modal_img_paths[0]).stem
#         lab_path = self._find_label_path_by_stem(stem)
#         w, h = self.args.load_width, self.args.load_height
#         imgs = []
#         for i in range(self.modal_num):
#             img_path_i = modal_img_paths[i]
#             img = safe_imread(img_path_i, cv2.IMREAD_UNCHANGED, desc=f"{self.modals[i]} image")
#             if img.ndim == 2:
#                 img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#             else:
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
#             imgs.append(img)
#         lab = safe_imread(lab_path, cv2.IMREAD_UNCHANGED, desc="label")
#         if lab.ndim == 3:
#             lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
#         if lab.dtype != np.uint8:
#             maxv = float(lab.max()) if lab.max() > 0 else 1.0
#             lab = (lab.astype(np.float32) / maxv * 255.0).astype(np.uint8)
#         lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_CUBIC)
#         _, lab = cv2.threshold(lab, 127, 255, cv2.THRESH_BINARY)
#         _, lab = cv2.threshold(lab, 127, 1, cv2.THRESH_BINARY)
#         imgs_transfer = []
#         for i in range(self.modal_num):
#             imgs_transfer.append(self.img_transforms(Image.fromarray(imgs[i].copy())))
#         lab_tensor = self.lab_transform(lab.copy()).unsqueeze(0)
#         returned = {}
#         for i in range(self.modal_num):
#             returned[self.modals[i]] = imgs_transfer[i]
#         returned['label'] = lab_tensor
#         returned['image_path'] = modal_img_paths[0]
#         returned['label_path'] = lab_path
#         return returned




import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from .base_dataset import BaseDataset
from .utils import MaskToTensor

try:
    import tifffile as tiff
    _HAS_TIFFFILE = True
except Exception:
    _HAS_TIFFFILE = False

IMG_EXTS = (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg")


def _exts_all_cases():
    return list(IMG_EXTS) + [e.upper() for e in IMG_EXTS]


def _list_images(dir_path: str, recursive: bool = False):
    p = Path(dir_path)
    if not p.exists():
        return []
    files = []
    exts = _exts_all_cases()
    if recursive:
        for ext in exts:
            files.extend(sorted(str(x) for x in p.rglob(f"*{ext}")))
    else:
        for ext in exts:
            files.extend(sorted(str(x) for x in p.glob(f"*{ext}")))
    return files


def _read_tiff(path: Path):
    arr = tiff.imread(str(path))
    if arr.ndim >= 3 and arr.shape[0] > 4 and arr.shape[-1] not in (1, 3, 4):
        arr = arr[0]
    return arr


def safe_imread(path, mode=cv2.IMREAD_UNCHANGED, desc="image"):

    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".tif", ".tiff") and _HAS_TIFFFILE:
        try:
            return _read_tiff(p)
        except Exception as e:
            raise FileNotFoundError(
                f"[DATASET] TIFF read error ({desc}): {p}\n"
                f"  - {e}\n"
                f"  - hint: consider converting to PNG/JPG if needed."
            )
    img = cv2.imread(str(p), mode)
    if img is None:
        raise FileNotFoundError(
            f"[DATASET] Unable to read {desc}: {p}\n"
            f"  - exists: {p.exists()}   is_file: {p.is_file()}\n"
            f"  - hint: check CWD/os.getcwd(), path spelling, case sensitivity, extension\n"
            f"  - note: for TIFF, installing 'tifffile' is recommended."
        )
    return img


def _normalize_to_01(img: np.ndarray, preferred_max: float = 0.0) -> np.ndarray:
    img = np.asarray(img)
    if img.dtype == np.bool_:
        return img.astype(np.float32)
    img = img.astype(np.float32, copy=False)
    if preferred_max and preferred_max > 0:
        if preferred_max <= 0:
            preferred_max = 1.0
        img = img / float(preferred_max)
        return np.clip(img, 0.0, 1.0, out=img)
    if img.dtype == np.float32 or img.dtype == np.float64:
        m = float(img.max()) if img.size > 0 else 1.0
        if m <= 1.000001:
            return img
        if m == 0:
            return img
        return (img / m).astype(np.float32)
    if img.dtype == np.uint8:
        return (img / 255.0).astype(np.float32)
    if img.dtype == np.uint16:
        return (img / 65535.0).astype(np.float32)
    m = float(img.max()) if img.size > 0 else 1.0
    if m == 0:
        return img
    return (img / m).astype(np.float32)


def _to_rgb01(img: np.ndarray, suffix: str, preferred_max: float = 0.0) -> np.ndarray:
    img = _normalize_to_01(img, preferred_max=preferred_max)
    img = np.squeeze(img)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)  # -> HxWx3
    elif img.ndim == 3:
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        elif img.shape[-1] >= 4:
            img = img[..., :3]
        else:
            pass
    else:
        img = np.squeeze(img)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[-1] >= 3:
            img = img[..., :3]
        else:
            raise ValueError(f"[DATASET] Unexpected image shape after squeeze: {img.shape}")

    if suffix not in (".tif", ".tiff"):
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img[..., ::-1].copy()
    img = img.astype(np.float32, copy=False)
    img = np.clip(img, 0.0, 1.0, out=img)
    return img


def _build_label_map(lab_dir: Path):
    if not lab_dir.exists():
        raise FileNotFoundError(f"[DATASET] Label directory not found: {lab_dir}")
    lab_map = {}
    exts = _exts_all_cases()
    for ext in exts:
        for p in lab_dir.glob(f"*{ext}"):
            lab_map[p.stem] = str(p)
    return lab_map


class CrackDataset(BaseDataset):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.modals = list(args.modals)
        self.modal_num = len(self.modals)
        self.preferred_max = float(getattr(args, "depth_max", 0.0))
        self.phase = getattr(args, "phase", "train")
        self.dataset_path = Path(args.dataset_path)
        modal_dirs = [self.dataset_path / f"{self.phase}_img_{m}" for m in self.modals]
        modal_stem2path = []
        for mdir, mname in zip(modal_dirs, self.modals):
            files = _list_images(mdir)
            if len(files) == 0:
                raise FileNotFoundError(
                    f"[DATASET] No images found for modal '{mname}' under {mdir}\n"
                    f"  - Expected files with extensions: {IMG_EXTS}"
                )
            s2p = {Path(f).stem: f for f in files}
            modal_stem2path.append(s2p)
        self.lab_dir = self.dataset_path / f"{self.phase}_lab"
        self.lab_map = _build_label_map(self.lab_dir)
        common = set(self.lab_map.keys())
        for s2p in modal_stem2path:
            common &= set(s2p.keys())
        if len(common) == 0:
            raise ValueError(
                "[DATASET] No common filenames among modals and labels.\n"
                f"  - modal0 samples: {len(modal_stem2path[0])}\n"
                f"  - modalN samples: {[len(s2p) for s2p in modal_stem2path[1:]]}\n"
                f"  - label samples : {len(self.lab_map)}\n"
                f"  - Please ensure consistent naming (same stem) across modals and labels."
            )
        self.stems = sorted(list(common))
        self.modal_img_paths = []
        for s2p in modal_stem2path:
            self.modal_img_paths.append([s2p[s] for s in self.stems])
        self.lab_transform = MaskToTensor()
        self.load_w = int(getattr(self.args, "load_width", 512))
        self.load_h = int(getattr(self.args, "load_height", 512))

    def __len__(self):
        return len(self.stems)

    def _find_label_path_by_stem(self, stem: str) -> str:
        p = self.lab_map.get(stem, None)
        if p is not None:
            return p
        exts = _exts_all_cases()
        for ext in exts:
            cand = self.lab_dir / f"{stem}{ext}"
            if cand.exists():
                self.lab_map[stem] = str(cand)
                return str(cand)
        raise FileNotFoundError(
            f"[DATASET] Label not found for stem='{stem}' under {self.lab_dir}\n"
            f"  - tried: {', '.join([stem + e for e in IMG_EXTS])}"
        )

    def __getitem__(self, index):
        if index < 0 or index >= len(self.stems):
            raise IndexError(f"[DATASET] Index out of range: {index}/{len(self.stems)}")
        stem = self.stems[index]
        lab_path = self._find_label_path_by_stem(stem)
        imgs_np = []
        for i in range(self.modal_num):
            img_path_i = self.modal_img_paths[i][index]
            suffix = Path(img_path_i).suffix.lower()
            img = safe_imread(img_path_i, cv2.IMREAD_UNCHANGED, desc=f"{self.modals[i]} image")
            img = _to_rgb01(img, suffix=suffix, preferred_max=self.preferred_max)
            img = cv2.resize(img, (self.load_w, self.load_h), interpolation=cv2.INTER_CUBIC)
            imgs_np.append(img)
        lab = safe_imread(lab_path, cv2.IMREAD_UNCHANGED, desc="label")
        lab = np.squeeze(lab)
        if lab.ndim == 3:
            if lab.shape[-1] == 3:
                lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
            elif lab.shape[-1] == 4:
                lab = cv2.cvtColor(lab, cv2.COLOR_BGRA2GRAY)
            else:
                lab = np.squeeze(lab)
        if lab.dtype != np.uint8:
            lab = _normalize_to_01(lab) * 255.0
        lab = lab.astype(np.uint8)
        lab = cv2.resize(lab, (self.load_w, self.load_h), interpolation=cv2.INTER_CUBIC)
        _, lab = cv2.threshold(lab, 127, 255, cv2.THRESH_BINARY)
        _, lab = cv2.threshold(lab, 127, 1, cv2.THRESH_BINARY)
        imgs_tensor = [torch.from_numpy(x.transpose(2, 0, 1)).float() for x in imgs_np]
        imgs_tensor = [(x - 0.5) / 0.5 for x in imgs_tensor]
        lab_tensor = self.lab_transform(lab.copy()).unsqueeze(0)
        returned = {self.modals[i]: imgs_tensor[i] for i in range(self.modal_num)}
        returned["label"] = lab_tensor
        returned["image_path"] = self.modal_img_paths[0][index]
        returned["label_path"] = lab_path
        return returned
