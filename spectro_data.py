#TODO: Ensure only 5% files are actually getting clipped before pad
import torch
import librosa
import numpy as np
import glob
import os
from tqdm import tqdm
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import IPython.display as ipd


HOP_LENGTH = 256
SR = 16000

#Used to determine max length to clip waveform data
def _computeMaxDuration():
    search_space = glob.glob("data/raw/**/*.wav", recursive=True)
    durations = [librosa.get_duration(path=f) for f in search_space]
    max_duration = np.percentile(durations, 95)
    return max_duration

#Create Spectrogram of Audio file
def _generateSpectrogram(file_path, max_duration, sr=SR, n_fft=512, hop_length=HOP_LENGTH):
    audio, _ = librosa.load(file_path, sr=sr)

    spectrogram = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    m_spectrogram = np.abs(spectrogram)

    # Apply padding to keep consistent shape
    max_time_frames = int(max_duration * sr / hop_length)

    # Handle 
    if m_spectrogram.shape[1] > max_time_frames:
        m_spectrogram = m_spectrogram[:, :max_time_frames]

    pad_spectrogram = np.pad(m_spectrogram, ((0, 0), (0, max_time_frames - m_spectrogram.shape[1])), mode="constant")

    return torch.from_numpy(librosa.util.normalize(pad_spectrogram))

#Create .pt Files for each Variant of Sound Data Type
def generateSpectrogramFiles(raw_data_path, output_path):
    if os.path.exists(output_path):
        print(f"Data already exists in {output_path}")
        data = torch.load(output_path)
        print(f"Data has shape: {data.shape}")
        return
    
    file_list = sorted(glob.glob(f"{raw_data_path}/*.wav"))
    max_duration = _computeMaxDuration()
    spectrograms = []
    print(f"Generating Spectrograms from: {raw_data_path} ... \n")
    for file in tqdm(file_list):
        spectrograms.append(_generateSpectrogram(file, max_duration))
    spec_tensor = torch.stack(spectrograms)
    torch.save(spec_tensor, output_path)
    print(f"Data Shape: {spec_tensor.shape}")

#Spectrogram to Visible Audio form
def specToAudio(spec_np, display=False, save_path=None):
    audio_waveform = librosa.istft(spec_np, hop_length=HOP_LENGTH)
    return ipd.Audio(audio_waveform, rate=SR)

#Loads pt files for clean and noisy respectively
class SpectrogramDataset(Dataset):
    def __init__(self, noisy_file, clean_file):
        self.clean_data = torch.load(clean_file)
        self.noisy_data = torch.load(noisy_file)

    def __len__(self):
        return len(self.noisy_data)
    
    def __getitem__(self, ind):
        noisy = self.noisy_data[ind].unsqueeze(0)
        clean = self.clean_data[ind].unsqueeze(0)
        
        frequency_bins = noisy.shape[1]
        #Clip spectrogram tensors to have consistent shape for training
        if frequency_bins % 2 == 1:
            noisy = noisy[:, :(frequency_bins-1), :]
            clean = clean[:, :(frequency_bins-1), :]
        
        time_frames = noisy.shape[2]
        if time_frames % 2 == 1:
            noisy = noisy[:, :, :(time_frames-1)]
            clean = clean[:, :, :(time_frames-1)]
        
        return noisy, clean
    
    def getSpec(self, ind, show=False):
        noisy, clean = self.__getitem__(ind)

        noisy_np = noisy.squeeze().numpy()
        clean_np = clean.squeeze().numpy()

        if show:
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            
            # Noisy spectrogram
            axs[0].imshow(noisy_np, aspect='auto', origin='lower')
            axs[0].set_title("Noisy Spectrogram")
            axs[0].set_xlabel("Time Frames")
            axs[0].set_ylabel("Frequency Bins")

            # Clean spectrogram
            axs[1].imshow(clean_np, aspect='auto', origin='lower')
            axs[1].set_title("Clean Spectrogram")
            axs[1].set_xlabel("Time Frames")
            
            plt.colorbar(axs[0].imshow(noisy_np, aspect='auto', origin='lower'), ax=axs[0])
            plt.colorbar(axs[1].imshow(clean_np, aspect='auto', origin='lower'), ax=axs[1])

            plt.show()

        return noisy_np, clean_np
        

    def playAudio(self, ind):
        pass

if __name__ == "__main__":
    generateSpectrogramFiles("data/raw/28spk/clean_sub/", "data/processed/28spk/clean_specs.pt")