import torchaudio
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import torch
from torchaudio.transforms import Spectrogram, Resample, MelSpectrogram, MelScale, AmplitudeToDB
from torchaudio.transforms import TimeStretch, FrequencyMasking, TimeMasking


class CusttomDataUrban:
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms
        self.pipeline = MyPipeline()

        label_list = sorted(list(set(df.iloc[:,-2])))
        self.label_dir = {k:v for v, k in enumerate(label_list)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = "/kaggle/input/urbansound8k/fold" + str(self.df["fold"].iloc[idx]) + "/" + self.df["slice_file_name"].iloc[idx]
        label = self.df["classID"].iloc[idx]

        wav, sr = torchaudio.load(path)

        target_sr = 44100  # Set your target sample rate here
        if sr != target_sr:
            wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
            sr = target_sr

        wav = torch.mean(wav, dim=0, keepdim=True)
        # return {"data": wav, "label": label}
        
        features = self.pipeline(wav)
        return {"data": features, "label": label}
    

class CusttomDataSnor:
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms
        self.pipeline = MyPipeline()
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

        # convert to mono
        audio_mono = torch.mean(wav, dim=0, keepdim=True)
        
        # pad 4 second
        tempData = torch.zeros([1, sr*4])
        if audio_mono.numel() < sr*4: # if sample_rate < 160000
            tempData[:, :audio_mono.numel()] = audio_mono
        else:
            tempData = audio_mono[:, :sr*4] # else sample_rate 160000
        audio_mono=tempData
        # return {"data": wav, "label": label}
        
        features = self.pipeline(audio_mono)
        return {"data": features, "label": label}
        
        # sample = {'data': wav, "sr":sr, 'label': label}
        # if self.transforms:
        #     sample = self.transforms(sample)
        # return sample
    

# class ToSpectrograms(object):
#     def __init__(self, n_fft, hop_length, n_mels):
#         self.n_fft = n_fft
#         self.hop_length = hop_length
#         self.n_mels = n_mels
        
#     def __call__(self, sample):
        
#         waveform, sr, label = sample['data'], sample['sr'], sample['label']
        
#         mel_spectrogram = T.MelSpectrogram(
#             sample_rate=sr,
#             n_fft=self.n_fft,
#             win_length=None,
#             hop_length=self.hop_length,
#             center=True,
#             pad_mode="reflect",
#             power=2.0,
#             norm="slaney",
#             n_mels=self.n_mels,
#             mel_scale="htk",
#         )
        
#         melspec = mel_spectrogram(waveform)
        
#         sample_new = {'data': melspec, 'label': label}
#         return sample_new
    
    
class MyPipeline(torch.nn.Module):
    def __init__(
        self,
        input_freq=44100,
        # resample_freq=8000,
        # n_fft=1024,
        # n_mel=128,
        stretch_factor=0.8,
    ):
        super().__init__()
        # self.resample = Resample(orig_freq=input_freq, new_freq=resample_freq)

        # self.spec = Spectrogram(n_fft=n_fft, power=2, win_length=None, hop_length=512)
        self.mel_spec = MelSpectrogram(
            sample_rate=input_freq,
                n_fft=1024,
                win_length=None,
                hop_length=512,
                center=True,
                pad_mode="reflect",
                power=2.0,
                norm='slaney',
                onesided=True,
                n_mels=128,
                mel_scale="htk",
            )
        
        self.spec_aug = torch.nn.Sequential(
            TimeStretch(stretch_factor, fixed_rate=True),
            FrequencyMasking(freq_mask_param=80),
            TimeMasking(time_mask_param=80),
        )

        # self.mel_scale = MelScale(
        #     n_mels=n_mel, sample_rate=input_freq, n_stft=n_fft // 2 + 1)

        self.log_scale = AmplitudeToDB(top_db=80)
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Resample the input
        # resampled = self.resample(waveform)

        # Convert to power spectrogram
        mel_spec = self.mel_spec(waveform)

        # Apply SpecAugment
        # spec = self.spec_aug(spec)

        # Convert to mel-scale
        # mel = self.mel_scale(spec)
        
        log_mel = self.log_scale(mel_spec)

        return log_mel   
    


# class MyPipeline(torch.nn.Module):
#     def __init__(
#         self,
#         input_freq=44100,
#         # resample_freq=44100,
#         n_fft=1024,
#         n_mels=256,
#         hop_length = 512,
#         stretch_factor=0.8,
#     ):
#         super().__init__()

#         # Resample the input waveform
#         # self.resample = Resample(orig_freq=input_freq, new_freq=resample_freq)

#         # Convert waveform to a power spectrogram
#         self.spec = MelSpectrogram(
#             sample_rate=input_freq,
#             n_fft=n_fft,
#             win_length=None,
#             hop_length=hop_length,
#             center=True,
#             pad_mode="reflect",
#             power=2.0,
#             norm="slaney",
#             n_mels=n_mels,
#             mel_scale="htk",
#         )

#         # Apply SpecAugment transformations
#         self.spec_aug = torch.nn.Sequential(
#             TimeStretch(stretch_factor, fixed_rate=True),
#             FrequencyMasking(freq_mask_param=80),
#             TimeMasking(time_mask_param=80),
#         )

#         # Convert the spectrogram to a mel-scale representation
#         # self.mel_scale = MelScale(
#         #     n_mels=n_mel, sample_rate=resample_freq, n_stft=n_fft // 2 + 1)

#     def forward(self, waveform: torch.Tensor) -> torch.Tensor:
#         # Resample the input waveform
#         resampled = self.resample(waveform)

#         # Convert the resampled waveform to a power spectrogram
#         spec = self.spec(resampled)

#         # Apply SpecAugment transformations
#         spec = self.spec_aug(spec)

#         # Convert the spectrogram to a mel-scale representation
#         mel = self.mel_scale(spec)

#         return mel


        
        
#     class MyPipeline(torch.nn.Module):
#     def __init__(
#         self,
#         input_freq=16000,
#         resample_freq=8000,
#         n_fft=1024,
#         n_mel=256,
#         stretch_factor=0.8,
#     ):
#         super().__init__()
#         self.resample = Resample(orig_freq=input_freq, new_freq=resample_freq)

#         self.spec = Spectrogram(n_fft=n_fft, power=2)

#         self.spec_aug = torch.nn.Sequential(
#             TimeStretch(stretch_factor, fixed_rate=True),
#             FrequencyMasking(freq_mask_param=80),
#             TimeMasking(time_mask_param=80),
#         )

#         self.mel_scale = MelScale(
#             n_mels=n_mel, sample_rate=resample_freq, n_stft=n_fft // 2 + 1)

#     def forward(self, waveform: torch.Tensor) -> torch.Tensor:
#         # Resample the input
#         resampled = self.resample(waveform)

#         # Convert to power spectrogram
#         spec = self.spec(resampled)

#         # Apply SpecAugment
#         spec = self.spec_aug(spec)

#         # Convert to mel-scale
#         mel = self.mel_scale(spec)

#         return mel
    

        
        
        
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

