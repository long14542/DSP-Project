import librosa
import librosa.display
import plotly.express as px
import numpy as np

audio_path1 = 'D:/20.27_-4-thg-6.wav'
audio_path2 = 'D:/20.27_-4-thg-6_2_.wav'

# Load audio files
audio1, sr1 = librosa.load(audio_path1)
audio2, sr2 = librosa.load(audio_path2)

# Compute the Short-Time Fourier Transform (STFT)
stft1 = librosa.stft(audio1)
stft2 = librosa.stft(audio2)

# Convert to spectrogram
spectrogram1 = np.abs(stft1)
spectrogram2 = np.abs(stft2)

# Compute Mel spectrogram
mel_spec1 = librosa.feature.melspectrogram(S=spectrogram1, sr=sr1)
mel_spec2 = librosa.feature.melspectrogram(S=spectrogram2, sr=sr2)

# Compute MFCCs
mfcc1 = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec1), n_mfcc=13)
mfcc2 = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec2), n_mfcc=13)
mfcc1 = np.zeros((13, 111))
mfcc1[:, :] = mfcc2[:, :111]

# Plot MFCCs using Plotly
fig = px.imshow(mfcc1, origin='lower', aspect='auto')
fig.update_layout(title='MFCC - Audio 1', xaxis_title='Time', yaxis_title='MFCC Coefficients')
fig.show()

fig = px.imshow(mfcc2, origin='lower', aspect='auto')
fig.update_layout(title='MFCC - Audio 2', xaxis_title='Time', yaxis_title='MFCC Coefficients')
fig.show()

# Combine the MFCCs into a single array
X = np.concatenate([mfcc1, mfcc2], axis=0)

# Create a target array indicating the class labels (e.g., 0 for audio 1, 1 for audio 2)
y = np.concatenate([np.zeros(mfcc1.shape[0]), np.ones(mfcc2.shape[0])], axis=0)
print(y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the SVM classifier with a linear kernel
clf = SVC(kernel='linear')

# Train the SVM classifier on the training data
clf.fit(X_train, y_train)