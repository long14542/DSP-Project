import librosa
import librosa.display
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

audio_path1 = '/Users/admin/Documents/GitHub/DSP-Project/recordings/co1.wav'
audio_path2 = '/Users/admin/Documents/GitHub/DSP-Project/recordings/khong1.wav'
audio_path3 = '/Users/admin/Documents/GitHub/DSP-Project/recordings/co2.wav'
audio_path4 = '/Users/admin/Documents/GitHub/DSP-Project/recordings/khong2.wav'
audio_path5 = '/Users/admin/Documents/GitHub/DSP-Project/recordings/co3.wav'
audio_path6 = '/Users/admin/Documents/GitHub/DSP-Project/recordings/khong3.wav'
audio_path7 = '/Users/admin/Documents/GitHub/DSP-Project/recordings/co4.wav'
audio_path8 = '/Users/admin/Documents/GitHub/DSP-Project/recordings/khong4.wav'
audio_path9 = '/Users/admin/Documents/GitHub/DSP-Project/recordings/co5.wav'
audio_path10 = '/Users/admin/Documents/GitHub/DSP-Project/recordings/khong5.wav'
audio_path11 = '/Users/admin/Documents/GitHub/DSP-Project/recordings/co6.wav'
audio_path12 = '/Users/admin/Documents/GitHub/DSP-Project/recordings/khong6.wav'
audio_path13 = '/Users/admin/Documents/GitHub/DSP-Project/recordings/co7.wav'
audio_path14 = '/Users/admin/Documents/GitHub/DSP-Project/recordings/khong7.wav'
audio_path15 = '/Users/admin/Documents/GitHub/DSP-Project/recordings/co8.wav'
audio_path16 = '/Users/admin/Documents/GitHub/DSP-Project/recordings/khong8.wav'
audio_path17 = '/Users/admin/Documents/GitHub/DSP-Project/recordings/co9.wav'
audio_path18 = '/Users/admin/Documents/GitHub/DSP-Project/recordings/khong9.wav'
audio_path19 = '/Users/admin/Documents/GitHub/DSP-Project/recordings/co10.wav'
audio_path20 = '/Users/admin/Documents/GitHub/DSP-Project/recordings/khong10.wav'

# Load audio files
audio1, sr1 = librosa.load(audio_path1)
audio2, sr2 = librosa.load(audio_path2)
audio3, sr3 = librosa.load(audio_path3)
audio4, sr4 = librosa.load(audio_path4)
audio5, sr5 = librosa.load(audio_path5)
audio6, sr6 = librosa.load(audio_path6)
audio7, sr7 = librosa.load(audio_path7)
audio8, sr8 = librosa.load(audio_path8)
audio9, sr9 = librosa.load(audio_path9)
audio10, sr10 = librosa.load(audio_path10)
audio11, sr11 = librosa.load(audio_path11)
audio12, sr12 = librosa.load(audio_path12)
audio13, sr13 = librosa.load(audio_path13)
audio14, sr14 = librosa.load(audio_path14)
audio15, sr15 = librosa.load(audio_path15)
audio16, sr16 = librosa.load(audio_path16)
audio17, sr17 = librosa.load(audio_path17)
audio18, sr18 = librosa.load(audio_path18)
audio19, sr19 = librosa.load(audio_path19)
audio20, sr20 = librosa.load(audio_path20)

# Compute the Short-Time Fourier Transform (STFT)
stft1 = librosa.stft(audio1)
stft2 = librosa.stft(audio2)
stft3 = librosa.stft(audio3)
stft4 = librosa.stft(audio4)
stft5 = librosa.stft(audio5)
stft6 = librosa.stft(audio6)

# Convert to spectrogram
spectrogram1 = np.abs(stft1)
spectrogram2 = np.abs(stft2)
spectrogram3 = np.abs(stft3)
spectrogram4 = np.abs(stft4)
spectrogram5 = np.abs(stft5)
spectrogram6 = np.abs(stft6)

# Compute Mel spectrogram
mel_spec1 = librosa.feature.melspectrogram(S=spectrogram1, sr=sr1)
mel_spec2 = librosa.feature.melspectrogram(S=spectrogram2, sr=sr2)
mel_spec3 = librosa.feature.melspectrogram(S=spectrogram3, sr=sr3)
mel_spec4 = librosa.feature.melspectrogram(S=spectrogram4, sr=sr4)
mel_spec5 = librosa.feature.melspectrogram(S=spectrogram5, sr=sr5)
mel_spec6 = librosa.feature.melspectrogram(S=spectrogram6, sr=sr6)
mel_spec7 = librosa.feature.melspectrogram(S=spectrogram1, sr=sr7)
mel_spec8 = librosa.feature.melspectrogram(S=spectrogram2, sr=sr8)
mel_spec9 = librosa.feature.melspectrogram(S=spectrogram3, sr=sr9)
mel_spec10 = librosa.feature.melspectrogram(S=spectrogram4, sr=sr10)
mel_spec11 = librosa.feature.melspectrogram(S=spectrogram5, sr=sr11)
mel_spec12 = librosa.feature.melspectrogram(S=spectrogram6, sr=sr12)
mel_spec13 = librosa.feature.melspectrogram(S=spectrogram1, sr=sr13)
mel_spec14 = librosa.feature.melspectrogram(S=spectrogram2, sr=sr14)
mel_spec15 = librosa.feature.melspectrogram(S=spectrogram3, sr=sr15)
mel_spec16 = librosa.feature.melspectrogram(S=spectrogram4, sr=sr16)
mel_spec17 = librosa.feature.melspectrogram(S=spectrogram5, sr=sr17)
mel_spec18 = librosa.feature.melspectrogram(S=spectrogram6, sr=sr18)
mel_spec19 = librosa.feature.melspectrogram(S=spectrogram5, sr=sr19)
mel_spec20 = librosa.feature.melspectrogram(S=spectrogram6, sr=sr20)

# Compute MFCCs
mfcc1 = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec1), n_mfcc=13)
mfcc2 = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec2), n_mfcc=13)
mfcc3 = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec3), n_mfcc=13)
mfcc4 = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec4), n_mfcc=13)
mfcc5 = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec5), n_mfcc=13)
mfcc6 = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec6), n_mfcc=13)
mfcc7 = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec7), n_mfcc=13)
mfcc8 = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec8), n_mfcc=13)
mfcc9 = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec9), n_mfcc=13)
mfcc10 = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec10), n_mfcc=13)
mfcc11 = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec11), n_mfcc=13)
mfcc12 = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec12), n_mfcc=13)
mfcc13 = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec13), n_mfcc=13)
mfcc14 = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec14), n_mfcc=13)
mfcc15 = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec15), n_mfcc=13)
mfcc16 = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec16), n_mfcc=13)
mfcc17 = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec17), n_mfcc=13)
mfcc18 = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec18), n_mfcc=13)
mfcc19 = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec19), n_mfcc=13)
mfcc20 = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec20), n_mfcc=13)

# Determine the maximum number of frames among the two arrays
max_frames = max(mfcc1.shape[1], mfcc2.shape[1], mfcc3.shape[1], mfcc4.shape[1], mfcc5.shape[1], mfcc6.shape[1], mfcc7.shape[1], mfcc8.shape[1], mfcc9.shape[1], mfcc10.shape[1], mfcc11.shape[1], mfcc12.shape[1], mfcc13.shape[1], mfcc14.shape[1], mfcc15.shape[1], mfcc16.shape[1], mfcc17.shape[1], mfcc18.shape[1], mfcc19.shape[1], mfcc20.shape[1])

# Pad mfcc if it has fewer frames than max_frames
def Pad(max_frames,mfcc):
    if mfcc.shape[1] < max_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])), mode='constant')
    return mfcc

mfcc1= Pad(max_frames, mfcc1)
mfcc2= Pad(max_frames, mfcc2)
mfcc3= Pad(max_frames, mfcc3)
mfcc4= Pad(max_frames, mfcc4)
mfcc5= Pad(max_frames, mfcc5)
mfcc6= Pad(max_frames, mfcc6)
mfcc7= Pad(max_frames, mfcc7)
mfcc8= Pad(max_frames, mfcc8)
mfcc9= Pad(max_frames, mfcc9)
mfcc10= Pad(max_frames, mfcc10)
mfcc11= Pad(max_frames, mfcc11)
mfcc12= Pad(max_frames, mfcc12)
mfcc13= Pad(max_frames, mfcc13)
mfcc14= Pad(max_frames, mfcc14)
mfcc15= Pad(max_frames, mfcc15)
mfcc16= Pad(max_frames, mfcc16)
mfcc17= Pad(max_frames, mfcc17)
mfcc18= Pad(max_frames, mfcc18)
mfcc19= Pad(max_frames, mfcc19)
mfcc20= Pad(max_frames, mfcc20)

# Plot MFCCs using Plotly
fig = px.imshow(mfcc1, origin='lower', aspect='auto')
fig.update_layout(title='MFCC - Audio 1', xaxis_title='Time', yaxis_title='MFCC Coefficients')
fig.show()

fig = px.imshow(mfcc2, origin='lower', aspect='auto')
fig.update_layout(title='MFCC - Audio 2', xaxis_title='Time', yaxis_title='MFCC Coefficients')
fig.show()

fig = px.imshow(mfcc3, origin='lower', aspect='auto')
fig.update_layout(title='MFCC - Audio 3', xaxis_title='Time', yaxis_title='MFCC Coefficients')
fig.show()

fig = px.imshow(mfcc4, origin='lower', aspect='auto')
fig.update_layout(title='MFCC - Audio 4', xaxis_title='Time', yaxis_title='MFCC Coefficients')
fig.show()

fig = px.imshow(mfcc5, origin='lower', aspect='auto')
fig.update_layout(title='MFCC - Audio 5', xaxis_title='Time', yaxis_title='MFCC Coefficients')
fig.show()

fig = px.imshow(mfcc6, origin='lower', aspect='auto')
fig.update_layout(title='MFCC - Audio 6', xaxis_title='Time', yaxis_title='MFCC Coefficients')
fig.show()

# Combine the MFCCs into a single array
X = np.concatenate([mfcc1, mfcc2, mfcc3, mfcc4, mfcc5, mfcc6, mfcc7, mfcc8, mfcc9, mfcc10, mfcc11, mfcc12, mfcc13, mfcc14, mfcc15, mfcc16, mfcc17, mfcc18, mfcc19, mfcc20], axis=0)

# Create a target array indicating the class labels (e.g., 0 for audio 1, 1 for audio 2)
y = np.concatenate([np.zeros(mfcc1.shape[0]), np.ones(mfcc2.shape[0]), np.zeros(mfcc3.shape[0]), np.ones(mfcc4.shape[0]), np.zeros(mfcc5.shape[0]), np.ones(mfcc6.shape[0]), np.zeros(mfcc7.shape[0]), np.ones(mfcc8.shape[0]), np.zeros(mfcc9.shape[0]), np.ones(mfcc10.shape[0]), np.zeros(mfcc11.shape[0]), np.ones(mfcc12.shape[0]), np.zeros(mfcc13.shape[0]), np.ones(mfcc14.shape[0]), np.zeros(mfcc15.shape[0]), np.ones(mfcc16.shape[0]), np.zeros(mfcc17.shape[0]), np.ones(mfcc18.shape[0]), np.zeros(mfcc19.shape[0]), np.ones(mfcc20.shape[0])], axis=0)
print(y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the SVM classifier with a linear kernel
clf = SVC(kernel='linear')

# Train the SVM classifier on the training data
clf.fit(X_train, y_train)

# Predict the labels for the testing data
y_pred = clf.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Compute precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Compute recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# Compute F1 score
f1 = f1_score(y_test, y_pred)
print("F1 score:", f1)