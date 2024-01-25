import torchaudio
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

class CusttomDataSnor:
    def __init__(self, df, transforms=None):
        self.df = df
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df["Link"].iloc[idx]
        label = int(self.df["Label"].iloc[idx])

        wav, sr = torchaudio.load(path)

        target_sr = 44100  # Set your target sample rate here
        if sr != target_sr:
            wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
            sr = target_sr


        wav = torch.mean(wav, dim=0, keepdim=True)
        
        sample = {'data': wav, "sr":sr, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample
    

class ToSpectrograms(object):
    def __init__(self, n_fft, hop_length, n_mels):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
    def __call__(self, sample):
        
        waveform, sr, label = sample['data'], sample['sr'], sample['label']
        
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=self.n_fft,
            win_length=None,
            hop_length=self.hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=self.n_mels,
            mel_scale="htk",
        )
        
        melspec = mel_spectrogram(waveform)
        
        sample_new = {'data': melspec, 'label': label}
        
        return sample_new
    
        
        
        
    

        
        
        
#         if self.transform:
# 			sample = self.transform(sample)
   
#    n_fft = 883
#         hop_length = 441
#         n_mels = 40
   
# 		return sample

#         mel_spectrogram = torchaudio.transforms.MelSpectrogram(
#             sample_rate=sr,
#             n_fft=n_fft,
#             hop_length=hop_length,
#             n_mels=n_mels
#         )(wav)

#         logmel = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

#         return logmel

