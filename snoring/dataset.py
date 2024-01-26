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
        # pad 4 second
        tempData = torch.zeros([1, sr*4])
        if wav.numel() < sr*4: # if sample_rate < 160000
            tempData[:, :wav.numel()] = wav
        else:
            tempData = wav[:, :sr*4] # else sample_rate 160000
        wav=tempData
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
    
class MyPipeline(torch.nn.Module):
    def __init__(
        self,
        input_freq=44100,
        stretch_factor=0.8,
    ):
        super().__init__()

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

        self.log_scale = AmplitudeToDB(top_db=80)
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel_spec = self.mel_spec(waveform)

        # Apply SpecAugment
        # spec = self.spec_aug(spec)

        log_mel = self.log_scale(mel_spec)

        return log_mel   
