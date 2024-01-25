# %%
from dataset import ToSpectrograms, CusttomDataSnor
import pandas as pd  
import torch 
from sklearn.model_selection import train_test_split
from glob import glob 
import matplotlib.pyplot as plt
# from utils import plot_spectrogram
import librosa.display
from lit import LitClassification
import lightning as pl
# %%

df = pd.DataFrame()
df["Link"] = glob("Snoring Dataset/*/*.wav")
df["Label"] = df["Link"].apply(lambda x: x.split("/")[-2])

df_train, df_test = train_test_split(df,
                        random_state=104,
                        test_size=0.2,
                        shuffle=True)

# transform = Tra


model_lit = LitClassification(CusttomDataSnor,df_train,df_test)

trainer = pl.Trainer(limit_train_batches=100)
trainer.fit(model_lit)

# data = CusttomDataSnor(df_train)







# %%
# data[0]['data'].shape

# %%
# import torchaudio.transforms as T
# mel_spectrogram = T.MelSpectrogram(
#     sample_rate=44100,
#     n_fft=1024,
#     win_length=None,
#     hop_length=512,
#     center=True,
#     pad_mode="reflect",
#     power=2.0,
#     norm='slaney',
#     onesided=True,
#     n_mels=128,
#     mel_scale="htk",
# )

# melspec = mel_spectrogram(data[11]["data"][0])
# plot_spectrogram(
#     melspec, title="MelSpectrogram - torchaudio", ylabel='mel freq')

# %%

# librosa.display.specshow(data[0]['data'][0].numpy())
# # plt.imshow(data[0]['data'][0])
# plt.savefig("okela.png")

# # %% 
# plt.hist(melspec.reshape(-1))