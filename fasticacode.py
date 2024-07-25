import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from tabulate import tabulate

# Generate mixed signals
np.random.seed(0)
n_samples = 1000
time = np.linspace(0, 8, n_samples)

# Source signals
s1 = np.sin(2 * time)  # Signal 1: sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2: square signal
s3 = np.random.randn(n_samples)  # Signal 3: random noise

S = np.c_[s1, s2, s3]  # Combine signals into a matrix

# Mix signals
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Mixed signals

# Calculate SNR before separation
signal_power = np.mean(np.square(S))
noise_power = np.mean(np.square(X - S))
snr_before = 10 * np.log10(signal_power / noise_power)

# Perform ICA
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)  # Reconstruct signals

# Calculate SNR after separation
signal_power_after = np.mean(np.square(S_))
noise_power_after = np.mean(np.square(X - S_))
snr_after = 10 * np.log10(signal_power_after / noise_power_after)

# Compute correlation coefficients
correlation_matrix = np.corrcoef(S.T, S_.T)

# Plot mixed and recovered signals
# Plot mixed signals
plt.figure(figsize=(12, 8))

plt.plot(X[:, 0], color='red', label='Mixed Signals')
plt.plot(X[:, 1], color='green', )

plt.title('Mixed Signals')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()

# Plot recovered signals
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(S_[:, 0], color='orange', label='Recovered Signal 1')
plt.title('Recovered Signal 1')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(S_[:, 1], color='purple', label='Recovered Signal 2')
plt.title('Recovered Signal 2')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()

# Plot correlation matrix
plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.title('Correlation Matrix')
plt.xlabel('Recovered Signals')
plt.ylabel('Original Sources')
plt.xticks(np.arange(2), ['Recovered Signal 1', 'Recovered Signal 2', ])
plt.yticks(np.arange(2), ['Original Source 1', 'Original Source 2', ])
plt.show()

# Print SNR before and after separation
print("SNR Before Separation:", snr_before, "dB")
print("SNR After Separation:", snr_after, "dB")
print("Noise Removed:", snr_before - snr_after, "dB")
headers = ['Recovered Signal 1', 'Recovered Signal 2']
print('Correlation matrix values obtained between original signal and recovered signal :')
table_data = [['Original Source 1'] + list(correlation_matrix[0]),
              ['Original Source 2'] + list(correlation_matrix[1]),
                                                                 ]
print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))