"""ICC profile building utilities.

This module provides minimal ICC v4 profile generation helpers for the GOG
(Gain-Offset-Gamma) model.
"""

from __future__ import annotations

import struct
from datetime import datetime

import numpy as np


def float_to_s15Fixed16(val: float) -> bytes:
    val = float(np.clip(val, -32768.0, 32767.0))
    fixed_val = int(round(val * 65536.0))
    return struct.pack(">i", fixed_val)


def write_xyz_number(xyz: np.ndarray) -> bytes:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
    return (
        float_to_s15Fixed16(float(xyz[0]))
        + float_to_s15Fixed16(float(xyz[1]))
        + float_to_s15Fixed16(float(xyz[2]))
    )


def write_xyz_tag(xyz: np.ndarray) -> bytes:
    return b"XYZ " + b"\x00" * 4 + write_xyz_number(xyz)


def write_sf32_tag(matrix_3x3: np.ndarray) -> bytes:
    m = np.asarray(matrix_3x3, dtype=np.float64).reshape(3, 3)
    payload = b"".join(float_to_s15Fixed16(v) for v in m.reshape(-1))
    return b"sf32" + b"\x00" * 4 + payload


def write_mluc_tag(text: str, language: str = "en", country: str = "US") -> bytes:
    if len(language) != 2 or len(country) != 2:
        raise ValueError("language/country must be 2 characters each, e.g. 'en'/'US'")

    s_bytes = text.encode("utf-16-be")

    num_records = 1
    record_size = 12
    header = b"mluc" + b"\x00" * 4 + struct.pack(">II", num_records, record_size)

    string_offset = 16 + num_records * record_size
    record = (
        language.encode("ascii")
        + country.encode("ascii")
        + struct.pack(">II", len(s_bytes), string_offset)
    )

    data = header + record + s_bytes
    padding = (4 - (len(data) % 4)) % 4
    return data + (b"\x00" * padding)


def bradford_adaptation_matrix(src_xyz: np.ndarray, dst_xyz: np.ndarray) -> np.ndarray:
    src = np.asarray(src_xyz, dtype=np.float64).reshape(3)
    dst = np.asarray(dst_xyz, dtype=np.float64).reshape(3)

    m = np.array(
        [
            [0.8951, 0.2664, -0.1614],
            [-0.7502, 1.7135, 0.0367],
            [0.0389, -0.0685, 1.0296],
        ],
        dtype=np.float64,
    )
    m_inv = np.linalg.inv(m)

    src_lms = m @ src
    dst_lms = m @ dst
    scale = np.diag(dst_lms / src_lms)
    return m_inv @ scale @ m


def write_para_tag(gamma_val: float, gain_val: float, offset_val: float) -> bytes:
    header = b"para" + b"\x00" * 4
    func_type = struct.pack(">H", 1)
    reserved = b"\x00" * 2

    params = (
        float_to_s15Fixed16(gamma_val)
        + float_to_s15Fixed16(gain_val)
        + float_to_s15Fixed16(offset_val)
    )

    return header + func_type + reserved + params


def generate_gog_icc_parametric(
    A: np.ndarray,
    k_vec: np.ndarray,
    b_vec: np.ndarray,
    g_vec: np.ndarray,
    description: str = "GOG Parametric Profile",
) -> bytes:
    trc_tags_data: list[bytes] = []
    for i in range(3):
        trc_tags_data.append(
            write_para_tag(float(g_vec[i]), float(k_vec[i]), float(b_vec[i]))
        )

    matrix_rows = [A[0], A[1], A[2]]

    d65_xyz = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    d50_xyz = np.array([0.96422, 1.0, 0.82521], dtype=np.float64)
    d65_to_d50 = bradford_adaptation_matrix(d65_xyz, d50_xyz)

    def _adapt_xyz_row_to_d50(xyz_row: np.ndarray) -> np.ndarray:
        xyz_row = np.asarray(xyz_row, dtype=np.float64).reshape(1, 3)
        return (xyz_row @ d65_to_d50.T).reshape(3)

    r_xyz = _adapt_xyz_row_to_d50(matrix_rows[0])
    g_xyz = _adapt_xyz_row_to_d50(matrix_rows[1])
    b_xyz = _adapt_xyz_row_to_d50(matrix_rows[2])

    tags: dict[str, bytes] = {
        "rXYZ": write_xyz_tag(r_xyz),
        "gXYZ": write_xyz_tag(g_xyz),
        "bXYZ": write_xyz_tag(b_xyz),
        "rTRC": trc_tags_data[0],
        "gTRC": trc_tags_data[1],
        "bTRC": trc_tags_data[2],
        "wtpt": write_xyz_tag(d50_xyz),
        "chad": write_sf32_tag(d65_to_d50),
        "cprt": write_mluc_tag("Copyright JackChou", "en", "US"),
        "desc": write_mluc_tag(description, "en", "US"),
    }

    header = bytearray(128)
    header[4:8] = b"lcms"
    header[8:12] = b"\x04\x30\x00\x00"
    header[12:16] = b"mntr"
    header[16:20] = b"RGB "
    header[20:24] = b"XYZ "

    now = datetime.now()
    struct.pack_into(
        ">6H",
        header,
        24,
        now.year,
        now.month,
        now.day,
        now.hour,
        now.minute,
        now.second,
    )
    header[36:40] = b"acsp"
    header[68:80] = write_xyz_number(d50_xyz)

    sorted_tags = sorted(tags.keys())
    num_tags = len(sorted_tags)
    tag_table = struct.pack(">I", num_tags)

    offset = 128 + 4 + (12 * num_tags)
    tag_data_concat = b""

    for tag in sorted_tags:
        data = tags[tag]
        size = len(data)
        padding_size = (4 - (size % 4)) % 4
        tag_table += tag.encode("ascii")
        tag_table += struct.pack(">II", offset, size)
        tag_data_concat += data + b"\x00" * padding_size
        offset += size + padding_size

    file_content = header + tag_table + tag_data_concat
    total_size = len(file_content)
    struct.pack_into(">I", file_content, 0, total_size)
    return bytes(file_content)


def generate_gog_icc_from_model(
    gog_model: dict,
    description: str = "GOG Model ICC Profile (Parametric)",
    assume_matrix_columns_are_primaries: bool = True,
) -> bytes:
    gain = np.asarray(gog_model["gain"], dtype=np.float64).reshape(3)
    offset = np.asarray(gog_model["offset"], dtype=np.float64).reshape(3)
    gamma = np.asarray(gog_model["gamma"], dtype=np.float64).reshape(3)
    matrix = np.asarray(gog_model["matrix"], dtype=np.float64).reshape(3, 3)

    if assume_matrix_columns_are_primaries:
        r_xyz = matrix[:, 0]
        g_xyz = matrix[:, 1]
        b_xyz = matrix[:, 2]
        A_rows = np.vstack([r_xyz, g_xyz, b_xyz])
    else:
        A_rows = matrix

    return generate_gog_icc_parametric(
        A_rows, gain, offset, gamma, description=description
    )
