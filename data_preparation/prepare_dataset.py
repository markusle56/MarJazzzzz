import random
from pathlib import Path
from typing import cast, Optional

import numpy as np
import symusic
import torch
from miditok import TokSequence
from torch.utils.data import DataLoader, Dataset

import featuriser as featuriser

MAX_SEQ_LEN  = 1024
BATCH_SIZE   = 32
NUM_FILES    = 2000
CSV_PATH     = "../filtered_midi.csv"
TRAIN_PATH   = "dataset_train.npy"
VAL_PATH     = "dataset_val.npy"
TEST_PATH    = "dataset_test.npy"
SPLIT        = (0.8, 0.1, 0.1)
RANDOM_SEED  = 42

# ── Augmentation Config ──────────────────────────────────────────────────────
PITCH_OFFSETS = list(range(-2, 3))
TIME_STRETCHES = [1.0]
VELOCITY_JITTER = 5                    # ±5 velocity randomisation
PITCH_MIN       = 30                 
PITCH_MAX       = 90                


def split_paths(
    paths: list[Path],
    train: float = SPLIT[0],
    val: float = SPLIT[1],
    seed: int = RANDOM_SEED,
) -> tuple[list[Path], list[Path], list[Path]]:
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    n       = len(paths)
    n_train = int(n * train)
    n_val   = int(n * val)
    return paths[:n_train], paths[n_train:n_train + n_val], paths[n_train + n_val:]


# ── Augmentation Functions ────────────────────────────────────────────────────

def transpose_score(score: symusic.Score, semitones: int) -> symusic.Score:
    """Transpose all notes, clamping to valid range instead of rejecting."""
    new_score = score.copy()
    for track in new_score.tracks:
        # Filter out notes that would go out of range
        track.notes = [n for n in track.notes 
                       if PITCH_MIN <= n.pitch + semitones < PITCH_MAX]
        for note in track.notes:
            note.pitch += semitones
    return new_score

def time_stretch_score(score: symusic.Score, factor: float) -> symusic.Score:
    """Stretch all note timings by the given factor."""
    if abs(factor - 1.0) < 0.001:
        return score
    new_score = score.copy()
    for track in new_score.tracks:
        for note in track.notes:
            note.time = int(note.time * factor)
            note.duration = max(1, int(note.duration * factor))
    for tempo in new_score.tempos:
        tempo.time = int(tempo.time * factor)
    return new_score


def jitter_velocity(score: symusic.Score, jitter: int) -> symusic.Score:
    """Add small random velocity variations to each note."""
    if jitter <= 0:
        return score
    new_score = score.copy()
    for track in new_score.tracks:
        for note in track.notes:
            offset = random.randint(-jitter, jitter)
            note.velocity = max(1, min(127, note.velocity + offset))
    return new_score


def augment_score(score: symusic.Score) -> list[symusic.Score]:
    """
    Generate augmented versions of a single score.
    Returns a list of valid augmented scores (including the original).
    """
    augmented = []
    for offset in PITCH_OFFSETS:
        transposed = transpose_score(score, offset)
        if transposed is None:
            continue  # Skip if transposition pushes notes out of range
        for stretch in TIME_STRETCHES:
            stretched = time_stretch_score(transposed, stretch)
            jittered = jitter_velocity(stretched, VELOCITY_JITTER)
            augmented.append(jittered)
    return augmented


# ── Tokenisation ──────────────────────────────────────────────────────────────

def tokenise_and_save(
    midi_paths: list[Path],
    max_seq_len: int = MAX_SEQ_LEN,
    out_path: str = TRAIN_PATH,
    augment: bool = False,
):
    """
    Tokenise MIDI files into fixed-length chunks and save as .npy.
    
    Args:
        midi_paths:  List of MIDI file paths
        max_seq_len: Length of each chunk
        out_path:    Output .npy file path
        augment:     If True, apply data augmentation (only for training set)
    """
    chunks: list[list[int]] = []
    failed = 0
    total_scores = 0

    for path in midi_paths:
        try:
            score = symusic.Score(str(path))
        except Exception:
            failed += 1
            continue

        # Get list of scores to tokenise (augmented or just original)
        if augment:
            scores = augment_score(score)
        else:
            scores = [score]

        for s in scores:
            try:
                ids: list[int] = cast(TokSequence, featuriser.tokenizer.encode(s)[0]).ids
            except Exception:
                continue

            total_scores += 1
            for start in range(0, len(ids) - max_seq_len + 1, max_seq_len):
                chunks.append(ids[start : start + max_seq_len])

    arr = np.array(chunks, dtype=np.int16)
    np.save(out_path, arr)
    print(f"Saved {len(chunks)} chunks from {total_scores} scores "
          f"({len(midi_paths) - failed} files) → '{out_path}'")
    print(f"Array shape: {arr.shape}  |  size on disk: {arr.nbytes / 1e6:.1f} MB")
    if failed:
        print(f"Skipped {failed} unreadable files")


# ── Dataset & DataLoader ─────────────────────────────────────────────────────

class PreTokenizedDataset(Dataset):
    """Load a pre-tokenized .npy file."""

    def __init__(self, path: str = TRAIN_PATH):
        self.data = torch.from_numpy(np.load(path).astype(np.int64))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.data[idx]}


def load_pretokenized(path: str = TRAIN_PATH, batch_size: int = BATCH_SIZE) -> DataLoader:
    return DataLoader(PreTokenizedDataset(path), batch_size=batch_size, shuffle=True)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    paths = featuriser.get_files(CSV_PATH, num_files=NUM_FILES)
    train_paths, val_paths, test_paths = split_paths(paths)

    print(f"Split: {len(train_paths)} train / {len(val_paths)} val / {len(test_paths)} test")

    # Augment ONLY training data — val and test stay clean
    tokenise_and_save(train_paths, out_path=TRAIN_PATH, augment=True)
    tokenise_and_save(val_paths,   out_path=VAL_PATH,   augment=False)
    tokenise_and_save(test_paths,  out_path=TEST_PATH,  augment=False)