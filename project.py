import librosa
import librosa.display
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

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

def Extracted_MFCC_features(audio_path):
    # Load audio files
    audio, sr = librosa.load(audio_path)
    # Compute the Short-Time Fourier Transform (STFT)
    stft = librosa.stft(audio)
    # Convert to spectrogram
    spectrogram = np.abs(stft)
    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(S=spectrogram, sr=sr)
    # Compute MFCCs
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=13)
    return mfcc

# Extracted MFCC features
mfcc1_original = Extracted_MFCC_features(audio_path1)
mfcc2_original = Extracted_MFCC_features(audio_path2)
mfcc3_original = Extracted_MFCC_features(audio_path3)
mfcc4_original = Extracted_MFCC_features(audio_path4)
mfcc5_original = Extracted_MFCC_features(audio_path5)
mfcc6_original = Extracted_MFCC_features(audio_path6)
mfcc7_original = Extracted_MFCC_features(audio_path7)
mfcc8_original = Extracted_MFCC_features(audio_path8)
mfcc9_original = Extracted_MFCC_features(audio_path9)
mfcc10_original = Extracted_MFCC_features(audio_path10)
mfcc11_original = Extracted_MFCC_features(audio_path11)
mfcc12_original = Extracted_MFCC_features(audio_path12)
mfcc13_original = Extracted_MFCC_features(audio_path13)
mfcc14_original = Extracted_MFCC_features(audio_path14)
mfcc15_original = Extracted_MFCC_features(audio_path15)
mfcc16_original = Extracted_MFCC_features(audio_path16)
mfcc17_original = Extracted_MFCC_features(audio_path17)
mfcc18_original = Extracted_MFCC_features(audio_path18)
mfcc19_original = Extracted_MFCC_features(audio_path19)
mfcc20_original = Extracted_MFCC_features(audio_path20)

# Determine the maximum number of frames among the two arrays(the number of frams depends on how long the recording is)
max_frames = 1000

# Pad mfcc if it has fewer frames than max_frames
def Pad(max_frames,mfcc):
    if mfcc.shape[1] < max_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])), mode='constant')
    return mfcc

mfcc1= Pad(max_frames, mfcc1_original)
mfcc2= Pad(max_frames, mfcc2_original)
mfcc3= Pad(max_frames, mfcc3_original)
mfcc4= Pad(max_frames, mfcc4_original)
mfcc5= Pad(max_frames, mfcc5_original)
mfcc6= Pad(max_frames, mfcc6_original)
mfcc7= Pad(max_frames, mfcc7_original)
mfcc8= Pad(max_frames, mfcc8_original)
mfcc9= Pad(max_frames, mfcc9_original)
mfcc10= Pad(max_frames, mfcc10_original)
mfcc11= Pad(max_frames, mfcc11_original)
mfcc12= Pad(max_frames, mfcc12_original)
mfcc13= Pad(max_frames, mfcc13_original)
mfcc14= Pad(max_frames, mfcc14_original)
mfcc15= Pad(max_frames, mfcc15_original)
mfcc16= Pad(max_frames, mfcc16_original)
mfcc17= Pad(max_frames, mfcc17_original)
mfcc18= Pad(max_frames, mfcc18_original)
mfcc19= Pad(max_frames, mfcc19_original)
mfcc20= Pad(max_frames, mfcc20_original)

# Plot 2 first audio
# Load audio files
audio1, sr = librosa.load(audio_path1, sr=None)
plt.figure()
plt.plot(audio1)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Audio 1 Signal (""Co"")")
plt.show()

# Load audio files
audio2, sr = librosa.load(audio_path2, sr=None)
plt.figure()
plt.plot(audio2)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Audio 2 Signal (""Khong"")")
plt.show()

# Plot 2 first MFCCs using Plotly
fig = px.imshow(mfcc1_original, origin='lower', aspect='auto')
fig.update_layout(title='MFCC - Audio 1 ("Co")', xaxis_title='Time', yaxis_title='MFCC Coefficients')
fig.show()

fig = px.imshow(mfcc2_original, origin='lower', aspect='auto')
fig.update_layout(title='MFCC - Audio 2 ("Khong")', xaxis_title='Time', yaxis_title='MFCC Coefficients')
fig.show()

# Combine the MFCCs into a single array
X = np.concatenate([mfcc1, mfcc2, mfcc3, mfcc4, mfcc5, mfcc6, mfcc7, mfcc8, mfcc9, mfcc10, mfcc11, mfcc12, mfcc13, mfcc14, mfcc15, mfcc16, mfcc17, mfcc18, mfcc19, mfcc20], axis=0)

# Create a target array indicating the class labels (e.g., 0 for "Co", 1 for "Khong")
y = np.concatenate([np.zeros(mfcc1.shape[0]), np.ones(mfcc2.shape[0]), np.zeros(mfcc3.shape[0]), np.ones(mfcc4.shape[0]), np.zeros(mfcc5.shape[0]), np.ones(mfcc6.shape[0]), np.zeros(mfcc7.shape[0]), np.ones(mfcc8.shape[0]), np.zeros(mfcc9.shape[0]), np.ones(mfcc10.shape[0]), np.zeros(mfcc11.shape[0]), np.ones(mfcc12.shape[0]), np.zeros(mfcc13.shape[0]), np.ones(mfcc14.shape[0]), np.zeros(mfcc15.shape[0]), np.ones(mfcc16.shape[0]), np.zeros(mfcc17.shape[0]), np.ones(mfcc18.shape[0]), np.zeros(mfcc19.shape[0]), np.ones(mfcc20.shape[0])], axis=0)
print(y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the SVM classifier with a linear kernel
clf = SVC(kernel='linear')

# Train the SVM classifier on the training data
clf.fit(X_train, y_train)

# Predict the labels for the testing data
y_pred = clf.predict(X_train)

# Compute accuracy
accuracy = accuracy_score(y_train, y_pred)
print("Accuracy:", accuracy)

# Compute precision
precision = precision_score(y_train, y_pred)
print("Precision:", precision)

# Compute recall
recall = recall_score(y_train, y_pred)
print("Recall:", recall)

# Compute F1 score
f1 = f1_score(y_train, y_pred)
print("F1 score:", f1)

def Apply_the_classifier(audio_path):
    # Extracted MFCC features
    mfcc = Extracted_MFCC_features(audio_path)
    #Pad mfcc1 so it has same features as trained SVM 
    mfcc = Pad(max_frames,mfcc)

    # Predict the label for the new instance
    predicted_label = clf.predict(mfcc)[0]

    # Convert the predicted label to "Co" or "Khong"
    if predicted_label == 0: label = "Co"
    elif predicted_label == 1: label = "Khong"
    else: label = "Unknow"

    # Print the predicted label
    print("Predicted Label:", label)

# Apply the classifier to predict the label:
n=0
while (n!=2):
    print("1. Choose audio file path")
    print("2. Exit")
    n = int(input("Enter your choice: "))
    if n==1:
        path = input("Enter your audio path: ")
        Apply_the_classifier(path)
