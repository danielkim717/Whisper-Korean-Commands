"""원본 cell 9·10의 오디오 로드 및 라벨 매핑 구현."""
import os
import numpy as np
import pandas as pd
import torch
import whisper
from torch.utils.data import Dataset
from .config import LABEL_MAP, LABEL_TO_ID
label_to_id = LABEL_TO_ID


class RawAudioDataset(Dataset):
    def __init__(self, data_df, target_length=48000):
        self.paths = data_df['path'].tolist()
        self.labels = data_df['label'].tolist()
        self.target_length = target_length

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]

        # 1. 오디오 로드 (Whisper load_audio는 16k 리샘플링 포함)
        audio = whisper.load_audio(path)

        # 2. 길이 맞추기 (Padding or Trimming) -> 고정 길이 48000
        if len(audio) > self.target_length:
            audio = audio[:self.target_length]
        else:
            audio = np.pad(audio, (0, self.target_length - len(audio)), mode='constant')

        # 3. Tensor 변환 (Float32)
        return torch.tensor(audio, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def create_metadata(data_dir):
    rows = []
    if not os.path.exists(data_dir):
        return pd.DataFrame(rows)

    for fn in os.listdir(data_dir):
        if fn.lower().endswith(".wav"):
            base = fn.split("_")[0]  # '1-1_xxx.wav' -> '1-1'
            if base in LABEL_MAP:
                rows.append({
                    "path": os.path.join(data_dir, fn),
                    "label": label_to_id[LABEL_MAP[base]]
                })
    return pd.DataFrame(rows)
