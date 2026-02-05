#!/usr/bin/env python3
import io
import math
import struct
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


SMPL_HEADER_FMT = "<9I"
SMPL_HEADER_SIZE = struct.calcsize(SMPL_HEADER_FMT)
SMPL_LOOP_FMT = "<6I"
SMPL_LOOP_SIZE = struct.calcsize(SMPL_LOOP_FMT)


def read_smpl_chunk(wav_bytes):
    if wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise ValueError("Not a RIFF/WAVE file")
    pos = 12
    while pos + 8 <= len(wav_bytes):
        chunk_id = wav_bytes[pos:pos + 4]
        size = struct.unpack_from("<I", wav_bytes, pos + 4)[0]
        data_start = pos + 8
        data_end = data_start + size
        if chunk_id == b"smpl":
            return wav_bytes[data_start:data_end]
        pos = data_end + (size % 2)
    return None


def parse_smpl(smpl_data):
    if smpl_data is None or len(smpl_data) < SMPL_HEADER_SIZE:
        return None
    header = struct.unpack_from(SMPL_HEADER_FMT, smpl_data, 0)
    num_loops = header[7]
    loops = []
    offset = SMPL_HEADER_SIZE
    for _ in range(num_loops):
        if offset + SMPL_LOOP_SIZE > len(smpl_data):
            break
        loops.append(struct.unpack_from(SMPL_LOOP_FMT, smpl_data, offset))
        offset += SMPL_LOOP_SIZE
    return header, loops


def build_smpl(header, loops, old_sr, new_sr, new_frames):
    if header is None:
        return None
    manufacturer, product, _sample_period, midi_unity, midi_pitch, smpte_fmt, smpte_off, num_loops, sampler_data = header
    sample_period = int(round(1_000_000_000 / new_sr))
    ratio = new_sr / old_sr
    new_loops = []
    for cue_id, loop_type, start, end, fraction, play_count in loops:
        new_start = int(round(start * ratio))
        new_end = int(round(end * ratio))
        new_start = max(0, min(new_start, new_frames - 1))
        new_end = max(0, min(new_end, new_frames - 1))
        new_loops.append((cue_id, loop_type, new_start, new_end, fraction, play_count))
    num_loops = len(new_loops)
    packed = struct.pack(
        SMPL_HEADER_FMT,
        manufacturer,
        product,
        sample_period,
        midi_unity,
        midi_pitch,
        smpte_fmt,
        smpte_off,
        num_loops,
        sampler_data,
    )
    for loop in new_loops:
        packed += struct.pack(SMPL_LOOP_FMT, *loop)
    return packed


def inject_chunk(wav_bytes, chunk_id, chunk_data):
    if chunk_data is None:
        return wav_bytes
    if len(chunk_id) != 4:
        raise ValueError("chunk_id must be 4 bytes")
    size = len(chunk_data)
    pad = b"\x00" if size % 2 else b""
    chunk = chunk_id + struct.pack("<I", size) + chunk_data + pad
    new_bytes = wav_bytes + chunk
    riff_size = len(new_bytes) - 8
    new_bytes = new_bytes[:4] + struct.pack("<I", riff_size) + new_bytes[8:]
    return new_bytes


def to_mono(data):
    if data.shape[1] == 1:
        return data
    return np.mean(data, axis=1, keepdims=True)


def resample_audio(data, old_sr, new_sr):
    if old_sr == new_sr:
        return data
    gcd = math.gcd(old_sr, new_sr)
    up = new_sr // gcd
    down = old_sr // gcd
    return resample_poly(data, up, down, axis=0).astype(np.float32, copy=False)


def bit_depth_to_subtype(bit_depth):
    mapping = {
        8: "PCM_U8",
        16: "PCM_16",
        24: "PCM_24",
        32: "PCM_32",
    }
    return mapping.get(bit_depth)


def apply_tpdf_dither(data, bit_depth):
    if bit_depth is None:
        return data
    lsb = 1.0 / (2 ** (bit_depth - 1))
    noise = (np.random.random(size=data.shape) - np.random.random(size=data.shape)) * lsb
    dithered = data + noise
    return np.clip(dithered, -1.0, 1.0)


def normalize_peak(data, target=0.999):
    peak = np.max(np.abs(data))
    if peak <= 0:
        return data
    return data * (target / peak)


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Process WAV preserving smpl loop points.")
    parser.add_argument("input", type=Path, help="Input WAV path")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output WAV path (default: <input>[_<rate>Hz][_mono][_8bit].wav)",
    )
    parser.add_argument(
        "--rate",
        type=int,
        help="Target sample rate in Hz (omit to keep original rate)",
    )
    parser.add_argument("--mono", action="store_true", help="Downmix to mono before resampling")
    parser.add_argument(
        "--bit-depth",
        type=int,
        choices=[8, 16, 24, 32],
        help="Output PCM bit depth (applies TPDF dither)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Peak-normalize to -0.01 dBFS before dither",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    if output_path is None:
        rate_flag_used = any(arg == "--rate" or arg.startswith("--rate=") for arg in sys.argv)
        mono_suffix = "_mono" if args.mono else ""
        bit_suffix = f"_{args.bit_depth}bit" if args.bit_depth else ""
        rate_suffix = f"_{args.rate}Hz" if rate_flag_used else ""
        output_path = input_path.with_name(
            f"{input_path.stem}{rate_suffix}{mono_suffix}{bit_suffix}{input_path.suffix}"
        )

    wav_bytes = input_path.read_bytes()
    smpl_data = read_smpl_chunk(wav_bytes)
    smpl_parsed = parse_smpl(smpl_data)

    info = sf.info(str(input_path))
    data, old_sr = sf.read(str(input_path), dtype="float32", always_2d=True)
    new_sr = args.rate if args.rate else old_sr
    if args.mono:
        data = to_mono(data)
    resampled = resample_audio(data, old_sr, new_sr)

    target_subtype = bit_depth_to_subtype(args.bit_depth) if args.bit_depth else info.subtype
    if args.normalize:
        resampled = normalize_peak(resampled, target=0.999)
    if args.bit_depth:
        resampled = apply_tpdf_dither(resampled, args.bit_depth)

    with io.BytesIO() as buf:
        sf.write(
            buf,
            resampled,
            new_sr,
            subtype=target_subtype,
            format="WAV",
        )
        wav_out = buf.getvalue()

    if smpl_parsed is not None:
        header, loops = smpl_parsed
        smpl_new = build_smpl(header, loops, old_sr, new_sr, resampled.shape[0])
        wav_out = inject_chunk(wav_out, b"smpl", smpl_new)

    output_path.write_bytes(wav_out)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
