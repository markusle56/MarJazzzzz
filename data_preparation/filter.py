import pandas as pd
import numpy as np
import os
import pretty_midi

ROOT = os.path.join(os.path.dirname(__file__), "midi_kong")

SOLO_PIANO_PROGRAMS = set(range(0, 8))  # Grand Piano through Clavinet


def filter_solo_piano(path: str) -> bool:
    """Keep only files where every non-drum track uses a piano program (0–7)."""
    try:
        midi = pretty_midi.PrettyMIDI(path)
    except Exception:
        return False
    if not midi.instruments:
        return False
    for inst in midi.instruments:
        if inst.is_drum:
            return False
        if inst.program not in SOLO_PIANO_PROGRAMS:
            return False
    return True


def filter_pitch_range(path: str, low: int = 30, high: int = 90, min_ratio: float = 0.70) -> bool:
    """Keep files where ≥70 % of notes fall within MIDI 40–90."""
    try:
        midi = pretty_midi.PrettyMIDI(path)
    except Exception:
        return False
    pitches = [n.pitch for inst in midi.instruments if not inst.is_drum for n in inst.notes]
    if not pitches:
        return False
    in_range = sum(1 for p in pitches if low <= p <= high)
    return in_range / len(pitches) >= min_ratio


def filter_tempo(path: str, bpm_low: float = 60.0, bpm_high: float = 140.0) -> bool:
    """Keep files whose estimated tempo falls within [bpm_low, bpm_high]."""
    try:
        midi = pretty_midi.PrettyMIDI(path)
    except Exception:
        return False
    _, tempos = midi.get_tempo_changes()
    if len(tempos) == 0:
        return False
    bpm = float(np.median(tempos))
    return bpm_low <= bpm <= bpm_high


def filter_duration(path: str, max_seconds: float = 600.0, min_seconds: float = 30.0) -> bool:
    """Discard files longer than 10 min and shorter than 30 sec."""
    try:
        midi = pretty_midi.PrettyMIDI(path)
    except Exception:
        return False
    return midi.get_end_time() <= max_seconds and midi.get_end_time() >= min_seconds


def filter_note_density(
    path: str,
    min_notes: int = 50,
    min_avg: float = 3.0,
    max_avg: float = 10.0,
) -> bool:
    """Discard files with fewer than min_notes notes, or avg note rate outside [min_avg, max_avg] notes/sec."""
    try:
        midi = pretty_midi.PrettyMIDI(path)
    except Exception:
        return False
    notes = [n for inst in midi.instruments if not inst.is_drum for n in inst.notes]
    if len(notes) < min_notes:
        return False
    duration = midi.get_end_time()
    if duration <= 0:
        return False
    avg = len(notes) / duration
    return min_avg <= avg <= max_avg


def count_files_per_folder(root: str) -> pd.DataFrame:
    rows = []
    for dirpath, _, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, os.path.dirname(root))
        depth = rel.count(os.sep)
        rows.append({
            "path": rel,
            "depth": depth,
            "file_count": len(filenames),
        })
    df = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
    return df


def build_filtered_csv(root: str, out_csv: str = "filtered_midi.csv") -> pd.DataFrame:
    """Walk root, apply all 3 filters, write passing paths to CSV, return DataFrame."""
    midi_files = [
        os.path.join(dirpath, f)
        for dirpath, _, filenames in os.walk(root)
        for f in filenames
        if f.lower().endswith((".mid", ".midi"))
    ]

    total = len(midi_files)
    print(f"Found {total} MIDI files — running filters...")

    passed = []
    for i, path in enumerate(midi_files, 1):
        if i % 100 == 0:
            print(f"  {i}/{total} checked, {len(passed)} passing so far")
        if not filter_solo_piano(path):
            continue
        if not filter_pitch_range(path):
            continue
        if not filter_tempo(path):
            continue
        if not filter_duration(path):
            continue
        if not filter_note_density(path):
            continue
        passed.append(path)

    df = pd.DataFrame({"path": passed})
    df.to_csv(out_csv, index=False)
    print(f"\n{len(passed)} / {total} files passed all filters → saved to '{out_csv}'")
    return df


if __name__ == "__main__":
    df = count_files_per_folder(ROOT)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", 80)
    print(df.to_string(index=False))
    print(f"\nTotal folders: {len(df)}")
    print(f"Total files:   {df['file_count'].sum()}")

    build_filtered_csv(ROOT)