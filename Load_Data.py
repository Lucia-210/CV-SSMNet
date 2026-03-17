import numpy as np
import os.path
from pathlib import Path
import scipy.io as sio


def _parse_envi_header(hdr_path: Path) -> dict:
    """
    Minimal ENVI header parser (enough for this repo's PolSARpro-exported single-band .bin files).
    We avoid external dependency on `spectral` so the project can run in restricted environments.
    """
    txt = Path(hdr_path).read_text(encoding="utf-8", errors="ignore")
    out: dict = {}
    for raw in txt.splitlines():
        line = raw.strip()
        if not line or line.lower() == "envi":
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        # Strip braces blocks (we don't need them for reading arrays)
        if v.startswith("{") and v.endswith("}"):
            v = v[1:-1].strip()
        out[k] = v
    return out


def _dtype_from_envi(data_type: int, byte_order: int) -> np.dtype:
    # ENVI data type codes (subset)
    base_map = {
        1: np.uint8,
        2: np.int16,
        3: np.int32,
        4: np.float32,
        5: np.float64,
        12: np.uint16,
        13: np.uint32,
        14: np.int64,
        15: np.uint64,
    }
    if data_type not in base_map:
        raise ValueError(f"Unsupported ENVI data type: {data_type}")
    dt = np.dtype(base_map[data_type])
    # byte order: 0=little endian, 1=big endian
    if byte_order == 0:
        return dt.newbyteorder("<")
    if byte_order == 1:
        return dt.newbyteorder(">")
    return dt


def _read_band_region(hdr_path, img_path, band_idx=0, crop=None):
    """
    Read an ENVI BSQ single-band float/int raster from `.hdr` + `.bin` (or other raw) file.
    This repo's datasets store each band as a standalone BSQ file with bands=1.
    """
    hdr_path = Path(hdr_path)
    img_path = Path(img_path)
    meta = _parse_envi_header(hdr_path)
    samples = int(meta.get("samples"))
    lines = int(meta.get("lines"))
    bands = int(meta.get("bands", 1))
    if bands != 1:
        # Not expected in this repo; keep implementation minimal and explicit.
        raise ValueError(f"Expected ENVI bands=1, got {bands} for {hdr_path}")
    header_offset = int(meta.get("header offset", 0))
    data_type = int(meta.get("data type"))
    byte_order = int(meta.get("byte order", 0))
    interleave = str(meta.get("interleave", "bsq")).lower()
    if interleave != "bsq":
        raise ValueError(f"Unsupported ENVI interleave '{interleave}' for {hdr_path} (expected bsq)")

    dtype = _dtype_from_envi(data_type, byte_order)
    # Use memmap for efficiency; slice if crop provided.
    mm = np.memmap(img_path, dtype=dtype, mode="r", offset=header_offset, shape=(lines, samples))
    if crop is None:
        return np.array(mm)
    r0, r1, c0, c1 = crop
    return np.array(mm[r0:r1, c0:c1])


def _load_labels_from_mat(mat_path, preferred_keys=None):
    """
    从 .mat 文件中自动加载标签数组。
    
    Args:
        mat_path: mat 文件路径
        preferred_keys: 优先尝试的键名列表，如 ['gt', 'GroundTruth', 'labels']
    
    Returns:
        labels: 标签数组
    """
    if preferred_keys is None:
        preferred_keys = ['gt', 'GroundTruth', 'groundtruth', 'labels', 'label', 'GT']
    
    mat_data = sio.loadmat(mat_path)
    
    # 首先尝试优先键名
    for key in preferred_keys:
        if key in mat_data:
            return mat_data[key]
    
    # 如果优先键名都不存在，自动查找第一个 2D 数组
    for key, value in mat_data.items():
        if key.startswith('__'):  # 跳过元数据键
            continue
        if isinstance(value, np.ndarray) and value.ndim == 2:
            print(f"  [自动检测] 使用键名 '{key}' 作为标签 (shape: {value.shape})")
            return value
    
    # 列出所有可用键名，帮助调试
    available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    raise KeyError(f"无法在 {mat_path} 中找到标签数组。可用键名: {available_keys}")


def load_data(name, crop_size=None):
    # Always resolve dataset paths relative to this file, not current working directory.
    # This prevents FileNotFoundError when scripts are launched from other folders.
    datasets_root = Path(__file__).resolve().parent / "Datasets"

    # -----------------------------
    # Dataset name aliases (case-insensitive)
    # -----------------------------
    if not isinstance(name, str):
        raise TypeError(f"dataset name must be str, got {type(name)}")

    raw_name = name
    name_l = name.strip().lower()

    # 兼容别名：部分脚本会传 'Flevoland'
    if name_l in ['flevoland', 'fl', 'fl_t']:
        name = 'FL_T'
    elif name_l in ['sanfrancisco', 'san_francisco', 'sf']:
        name = 'SF'
    elif name_l in ['oberpfaffenhofen', 'ober', 'op', 'ober_t3']:
        # 默认返回 T3 子集（6通道），保持与其他T3数据集一致
        name = 'ober'
    elif name_l in ['ober_t6', 'oberpfaffenhofen_t6', 'op_t6']:
        # 返回完整 T6（21通道）
        name = 'ober_t6'
    else:
        name = raw_name

    if name == 'FL_T':
        path = datasets_root / 'Flevoland' / 'T3'
        meta = _parse_envi_header(path / 'T11.bin.hdr')
        rows, cols = int(meta.get("lines")), int(meta.get("samples"))

        # 计算裁剪区域
        if crop_size is not None:
            h, w = crop_size
            h = min(h, rows)
            w = min(w, cols)
            r0 = (rows - h) // 2
            c0 = (cols - w) // 2
            crop = (r0, r0 + h, c0, c0 + w)
            T = np.zeros((h, w, 6), dtype=np.complex64)
        else:
            crop = None
            T = np.zeros((rows, cols, 6), dtype=np.complex64)

        T[:, :, 0] = _read_band_region(path / 'T11.bin.hdr', path / 'T11.bin', 0, crop)
        T[:, :, 1] = _read_band_region(path / 'T22.bin.hdr', path / 'T22.bin', 0, crop)
        T[:, :, 2] = _read_band_region(path / 'T33.bin.hdr', path / 'T33.bin', 0, crop)
        T[:, :, 3] = _read_band_region(path / 'T12_real.bin.hdr', path / 'T12_real.bin', 0, crop) + \
                1j * _read_band_region(path / 'T12_imag.bin.hdr', path / 'T12_imag.bin', 0, crop)
        T[:, :, 4] = _read_band_region(path / 'T13_real.bin.hdr', path / 'T13_real.bin', 0, crop) + \
                1j * _read_band_region(path / 'T13_imag.bin.hdr', path / 'T13_imag.bin', 0, crop)
        T[:, :, 5] = _read_band_region(path / 'T23_real.bin.hdr', path / 'T23_real.bin', 0, crop) + \
                1j * _read_band_region(path / 'T23_imag.bin.hdr', path / 'T23_imag.bin', 0, crop)

        labels = _load_labels_from_mat(datasets_root / 'Flevoland' / 'Flevoland_gt.mat')
        if crop is not None:
            labels = labels[crop[0]:crop[1], crop[2]:crop[3]]
##############################################################################

    elif name == 'SF':
        path = datasets_root / 'san_francisco' / 'T3'
        meta = _parse_envi_header(path / 'T11.bin.hdr')
        rows, cols = int(meta.get("lines")), int(meta.get("samples"))
        if crop_size is not None:
            h, w = crop_size
            h = min(h, rows)
            w = min(w, cols)
            r0 = (rows - h) // 2
            c0 = (cols - w) // 2
            crop = (r0, r0 + h, c0, c0 + w)
            T = np.zeros((h, w, 6), dtype=np.complex64)
        else:
            crop = None
            T = np.zeros((rows, cols, 6), dtype=np.complex64)

        T[:, :, 0] = _read_band_region(path / 'T11.bin.hdr', path / 'T11.bin', 0, crop)
        T[:, :, 1] = _read_band_region(path / 'T22.bin.hdr', path / 'T22.bin', 0, crop)
        T[:, :, 2] = _read_band_region(path / 'T33.bin.hdr', path / 'T33.bin', 0, crop)
        T[:, :, 3] = _read_band_region(path / 'T12_real.bin.hdr', path / 'T12_real.bin', 0, crop) + \
                1j * _read_band_region(path / 'T12_imag.bin.hdr', path / 'T12_imag.bin', 0, crop)
        T[:, :, 4] = _read_band_region(path / 'T13_real.bin.hdr', path / 'T13_real.bin', 0, crop) + \
                1j * _read_band_region(path / 'T13_imag.bin.hdr', path / 'T13_imag.bin', 0, crop)
        T[:, :, 5] = _read_band_region(path / 'T23_real.bin.hdr', path / 'T23_real.bin', 0, crop) + \
                1j * _read_band_region(path / 'T23_imag.bin.hdr', path / 'T23_imag.bin', 0, crop)

        labels = _load_labels_from_mat(datasets_root / 'san_francisco' / 'SanFrancisco_gt.mat')
        if crop is not None:
            labels = labels[crop[0]:crop[1], crop[2]:crop[3]]

##############################################################################
    elif name in ['ober', 'ober_t6']:
        path = datasets_root / 'Oberpfaffenhofen' / 'ESAR_Oberpfaffenhofen_T6'
        meta = _parse_envi_header(path / 'T11.bin.hdr')
        rows, cols = int(meta.get("lines")), int(meta.get("samples"))
        if crop_size is not None:
            h, w = crop_size
            h = min(h, rows)
            w = min(w, cols)
            r0 = (rows - h) // 2
            c0 = (cols - w) // 2
            crop = (r0, r0 + h, c0, c0 + w)
            T = np.zeros((h, w, 21), dtype=np.complex64)
        else:
            crop = None
            T = np.zeros((rows, cols, 21), dtype=np.complex64)

        T[:, :, 0] = _read_band_region(path / 'T11.bin.hdr', path / 'T11.bin', 0, crop)
        T[:, :, 1] = _read_band_region(path / 'T22.bin.hdr', path / 'T22.bin', 0, crop)
        T[:, :, 2] = _read_band_region(path / 'T33.bin.hdr', path / 'T33.bin', 0, crop)
        T[:, :, 3] = _read_band_region(path / 'T44.bin.hdr', path / 'T44.bin', 0, crop)
        T[:, :, 4] = _read_band_region(path / 'T55.bin.hdr', path / 'T55.bin', 0, crop)
        T[:, :, 5] = _read_band_region(path / 'T66.bin.hdr', path / 'T66.bin', 0, crop)
        T[:, :, 6] = _read_band_region(path / 'T12_real.bin.hdr', path / 'T12_real.bin', 0, crop) + \
                1j * _read_band_region(path / 'T12_imag.bin.hdr', path / 'T12_imag.bin', 0, crop)
        T[:, :, 7] = _read_band_region(path / 'T13_real.bin.hdr', path / 'T13_real.bin', 0, crop) + \
                1j * _read_band_region(path / 'T13_imag.bin.hdr', path / 'T13_imag.bin', 0, crop)
        T[:, :, 8] = _read_band_region(path / 'T14_real.bin.hdr', path / 'T14_real.bin', 0, crop) + \
                1j * _read_band_region(path / 'T14_imag.bin.hdr', path / 'T14_imag.bin', 0, crop)
        T[:, :, 9] = _read_band_region(path / 'T15_real.bin.hdr', path / 'T15_real.bin', 0, crop) + \
                1j * _read_band_region(path / 'T15_imag.bin.hdr', path / 'T15_imag.bin', 0, crop)
        T[:, :, 10] = _read_band_region(path / 'T16_real.bin.hdr', path / 'T16_real.bin', 0, crop) + \
                 1j * _read_band_region(path / 'T16_imag.bin.hdr', path / 'T16_imag.bin', 0, crop)
        T[:, :, 11] = _read_band_region(path / 'T23_real.bin.hdr', path / 'T23_real.bin', 0, crop) + \
                 1j * _read_band_region(path / 'T23_imag.bin.hdr', path / 'T23_imag.bin', 0, crop)
        T[:, :, 12] = _read_band_region(path / 'T24_real.bin.hdr', path / 'T24_real.bin', 0, crop) + \
                 1j * _read_band_region(path / 'T24_imag.bin.hdr', path / 'T24_imag.bin', 0, crop)
        T[:, :, 13] = _read_band_region(path / 'T25_real.bin.hdr', path / 'T25_real.bin', 0, crop) + \
                 1j * _read_band_region(path / 'T25_imag.bin.hdr', path / 'T25_imag.bin', 0, crop)
        T[:, :, 14] = _read_band_region(path / 'T26_real.bin.hdr', path / 'T26_real.bin', 0, crop) + \
                 1j * _read_band_region(path / 'T26_imag.bin.hdr', path / 'T26_imag.bin', 0, crop)
        T[:, :, 15] = _read_band_region(path / 'T34_real.bin.hdr', path / 'T34_real.bin', 0, crop) + \
                 1j * _read_band_region(path / 'T34_imag.bin.hdr', path / 'T34_imag.bin', 0, crop)
        T[:, :, 16] = _read_band_region(path / 'T35_real.bin.hdr', path / 'T35_real.bin', 0, crop) + \
                 1j * _read_band_region(path / 'T35_imag.bin.hdr', path / 'T35_imag.bin', 0, crop)
        T[:, :, 17] = _read_band_region(path / 'T36_real.bin.hdr', path / 'T36_real.bin', 0, crop) + \
                 1j * _read_band_region(path / 'T36_imag.bin.hdr', path / 'T36_imag.bin', 0, crop)
        T[:, :, 18] = _read_band_region(path / 'T45_real.bin.hdr', path / 'T45_real.bin', 0, crop) + \
                 1j * _read_band_region(path / 'T45_imag.bin.hdr', path / 'T45_imag.bin', 0, crop)
        T[:, :, 19] = _read_band_region(path / 'T46_real.bin.hdr', path / 'T46_real.bin', 0, crop) + \
                 1j * _read_band_region(path / 'T46_imag.bin.hdr', path / 'T46_imag.bin', 0, crop)
        T[:, :, 20] = _read_band_region(path / 'T56_real.bin.hdr', path / 'T56_real.bin', 0, crop) + \
                 1j * _read_band_region(path / 'T56_imag.bin.hdr', path / 'T56_imag.bin', 0, crop)

        labels = _load_labels_from_mat(datasets_root / 'Oberpfaffenhofen' / 'Oberpfaffenhofen_gt.mat')
        if crop is not None:
            labels = labels[crop[0]:crop[1], crop[2]:crop[3]]

        # 默认返回 T3 子集（6通道），与 Flevoland / SF 的 T3 形式保持一致：
        # [T11, T22, T33, T12, T13, T23]
        # 在 T6(21通道) 中对应索引：
        # T11=0, T22=1, T33=2, T12=6, T13=7, T23=11
        if name == 'ober':
            T3 = np.zeros((T.shape[0], T.shape[1], 6), dtype=np.complex64)
            T3[:, :, 0] = T[:, :, 0]
            T3[:, :, 1] = T[:, :, 1]
            T3[:, :, 2] = T[:, :, 2]
            T3[:, :, 3] = T[:, :, 6]
            T3[:, :, 4] = T[:, :, 7]
            T3[:, :, 5] = T[:, :, 11]
            T = T3
##############################################################################
    elif name == 'GroundFQ13':
        path = datasets_root / 'GroundFQ13' / 'T3'
        meta = _parse_envi_header(path / 'T11.bin.hdr')
        rows, cols = int(meta.get("lines")), int(meta.get("samples"))

        # 计算裁剪区域
        if crop_size is not None:
            h, w = crop_size
            h = min(h, rows)
            w = min(w, cols)
            r0 = (rows - h) // 2
            c0 = (cols - w) // 2
            crop = (r0, r0 + h, c0, c0 + w)
            T = np.zeros((h, w, 6), dtype=np.complex64)
        else:
            crop = None
            T = np.zeros((rows, cols, 6), dtype=np.complex64)

        T[:, :, 0] = _read_band_region(path / 'T11.bin.hdr', path / 'T11.bin', 0, crop)
        T[:, :, 1] = _read_band_region(path / 'T22.bin.hdr', path / 'T22.bin', 0, crop)
        T[:, :, 2] = _read_band_region(path / 'T33.bin.hdr', path / 'T33.bin', 0, crop)
        T[:, :, 3] = _read_band_region(path / 'T12_real.bin.hdr', path / 'T12_real.bin', 0, crop) + \
                1j * _read_band_region(path / 'T12_imag.bin.hdr', path / 'T12_imag.bin', 0, crop)
        T[:, :, 4] = _read_band_region(path / 'T13_real.bin.hdr', path / 'T13_real.bin', 0, crop) + \
                1j * _read_band_region(path / 'T13_imag.bin.hdr', path / 'T13_imag.bin', 0, crop)
        T[:, :, 5] = _read_band_region(path / 'T23_real.bin.hdr', path / 'T23_real.bin', 0, crop) + \
                1j * _read_band_region(path / 'T23_imag.bin.hdr', path / 'T23_imag.bin', 0, crop)

        # GroundFQ13 的标签键名是 'GroundTruth'
        labels = _load_labels_from_mat(
            datasets_root / 'GroundFQ13' / 'GroundFQ13.mat',
            preferred_keys=['GroundTruth', 'gt', 'labels']
        )
        if crop is not None:
            labels = labels[crop[0]:crop[1], crop[2]:crop[3]]
##############################################################################

    else:
        raise ValueError(
            f"Incorrect data name: {name}. Supported: FL_T (Flevoland), SF (SanFrancisco), "
            f"ober (Oberpfaffenhofen T3-subset), ober_t6 (Oberpfaffenhofen full T6), GroundFQ13"
        )

    return T, labels
