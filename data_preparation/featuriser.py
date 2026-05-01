from pathlib import Path
from typing import cast

import pandas as pd
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DataCollator, DatasetMIDI

TOKENIZER_PARAMS = {
    "pitch_range": (40, 90),
    "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
    "tempo_range": (80, 160),
    "use_pitchdrum_tokens": False,
}

config = TokenizerConfig(**TOKENIZER_PARAMS)
tokenizer = REMI(config)

def get_files(csv_address, num_files):
    df = pd.read_csv(csv_address)
    num_files = min(num_files, len(df))
    paths = df["path"].sample(n=num_files, random_state=42).tolist()
    return [Path(p) for p in paths]

def tokenise(midipaths, tokenizer, max_seq_len=1024):
    dataset = DatasetMIDI(
        files_paths=midipaths,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        bos_token_id=cast(int, tokenizer["BOS_None"]),
        eos_token_id=cast(int, tokenizer["EOS_None"]),
    )
    collator = DataCollator(tokenizer.pad_token_id)
    return dataset, collator

