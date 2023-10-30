import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

class VocosDataset(Dataset):
    def __init__(self, filelist_path: str, sampling_rate: int, num_samples: int, train: bool):
        with open(filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = sampling_rate
        self.num_samples = num_samples
        self.train = train

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int) -> torch.Tensor:
        audio_path = self.filelist[index]
        y, sr = torchaudio.load(audio_path)
        if y.size(0) > 1:
            # mix to mono
            y = y.mean(dim=0, keepdim=True)

        #gain = np.random.uniform(-1, -6) if self.train else -3
        #y = normalize(y, gain)

        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
        if y.size(-1) < self.num_samples:
            pad_length = self.num_samples - y.size(-1)
            padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
            y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
        elif self.train:
            start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
            y = y[:, start : start + self.num_samples]
        else:
            # During validation, take always the first segment for determinism
            y = y[:, : self.num_samples]

        return y[0]