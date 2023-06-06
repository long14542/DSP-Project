"""The provided code extracts Mel-Frequency Cepstral Coefficients (MFCC) from two speech audio files using the librosa library in Python. The MFCCs are then visualized using the 
plotly library.
The code begins by importing the necessary libraries: librosa, librosa.display, plotly.express, and numpy. These libraries are used for audio processing, visualization, and 
numerical computations.
The paths to the audio files are specified using the variables audio_path1 and audio_path2. The librosa.load() function is used to load the audio files and obtain the audio 
waveforms (audio1 and audio2) and the respective sampling rates (sr1 and sr2).
Next, the Short-Time Fourier Transform (STFT) is computed for each audio waveform using the librosa.stft() function. The resulting STFTs are stored in the variables stft1 and 
stft2.
To create spectrograms, the magnitudes of the STFTs are calculated using np.abs() and stored in the variables spectrogram1 and spectrogram2.
The Mel spectrograms are then computed using the librosa.feature.melspectrogram() function. The spectrograms (spectrogram1 and spectrogram2) and the corresponding sampling rates 
(sr1 and sr2) are provided as inputs to this function. The resulting Mel spectrograms are stored in the variables mel_spec1 and mel_spec2.
Finally, the MFCCs are calculated from the Mel spectrograms using the librosa.feature.mfcc() function. The power spectrograms obtained from librosa.power_to_db() are passed as 
inputs, along with the desired number of MFCC coefficients (n_mfcc=13). The computed MFCCs are stored in the variables mfcc1 and mfcc2.
To visualize the MFCCs, the px.imshow() function from plotly.express is used. Two separate figures are created to display the MFCCs of audio 1 and audio 2. The MFCC matrices 
(mfcc1 and mfcc2) are passed as inputs to px.imshow(), and the plot is customized with titles, x-axis labels, and y-axis labels using the update_layout() method. The resulting 
figures are displayed using the fig.show() function. We got MFCC Audio 1 & 2"""

