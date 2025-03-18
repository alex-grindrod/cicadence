import torch
import torch.nn.functional as F
import librosa
import numpy as np
import glob
import os
import torchaudio
import gc
from tqdm import tqdm
from torch.utils.data import Dataset

RAW_DATA_PATH = "data/raw/28spk/"
torch.manual_seed(42) #Consistent results

class WaveformDataset(Dataset):
    def __init__(self, noisy_waves, clean_waves, fraction=1):
        self.clean_data = torch.load(clean_waves)
        self.noisy_data = torch.load(noisy_waves)

        total_samples = len(self.clean_data)
        selected_samples = int(total_samples * fraction)

        self.clean_data = self.clean_data[:selected_samples]
        self.noisy_data = self.noisy_data[:selected_samples]

    def __len__(self):
        return len(self.clean_data)
    
    def __getitem__(self, ind):
        return self.noisy_data[ind].unsqueeze(0), self.clean_data[ind].unsqueeze(0)

    
if __name__ == "__main__":
    # Manually do before running:
    clean_save_path = "data/processed/28spk/combined_clean_waves.pt"
    noisy_save_path = "data/processed/28spk/combined_noisy_waves.pt"
    if os.path.exists(clean_save_path) or os.path.exists(noisy_save_path):
        print("At least one data file already exists --> comment out this if statement to overwrite files")
        print("You can also just delete combined_noisy_waves.pt and combined_clean_waves.pt")
        exit()

    clean_dir = os.path.join(RAW_DATA_PATH, "clean_train")
    noisy_dir = os.path.join(RAW_DATA_PATH, "noisy_train")

    for cfile, nfile in zip(os.listdir(clean_dir), os.listdir(noisy_dir)):
        if cfile != nfile:
            print("ERROR Non-Matching Files: ", end='')
            print(cfile, nfile)
            print("Dataset download went wrong probably")
    gc.collect()
    
    clean_waves = []
    noisy_waves = []
    i = 0
    print("\nExtracting waveforms: \n")
    for cfile, nfile in tqdm(zip(os.listdir(clean_dir), os.listdir(noisy_dir))):
        if ".wav" not in cfile:
            continue
        clean_waves.append(torchaudio.load(os.path.join(clean_dir, cfile))[0])
        noisy_waves.append(torchaudio.load(os.path.join(noisy_dir, cfile))[0])
    
    print("Concatenating and uniformly splitting data... \n")
    all_clean = torch.cat(clean_waves, dim=1)
    gc.collect()
    all_noisy = torch.cat(noisy_waves, dim=1)
    gc.collect()

    sample_size = 262144
    num_samples = all_clean.shape[1] // sample_size
    clean_clip = all_clean[:, :num_samples * sample_size]
    noisy_clip = all_noisy[:, :num_samples * sample_size]
    clean_samples = clean_clip.reshape(1, num_samples, sample_size).squeeze(0)
    noisy_samples = noisy_clip.reshape(1, num_samples, sample_size).squeeze(0)

    print("Saving Data to Dir ... \n")
    torch.save(clean_samples, clean_save_path)    
    torch.save(noisy_samples, noisy_save_path)
    print("Done!")
