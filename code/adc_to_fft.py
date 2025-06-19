import os
import numpy as np
import pandas as pd
from scipy import fftpack

present_dir = "/Users/kushalprakash/Documents/Uni/23-24/AIS+ML/WithBaby"
absent_dir = "/Users/kushalprakash/Documents/Uni/23-24/AIS+ML/WithoutBaby" 

chunk_size = 160000


def process_data(file, directory, output_filename):
  """
  Processes a single CSV file and saves the FFT magnitude to a new CSV.
  """
  if not file.endswith('.csv'):
    return

  file_path = os.path.join(directory, file)
  reader = pd.read_csv(file_path, chunksize=chunk_size)

  for chunk in reader:
    adc_data = chunk.iloc[:, 16:].values
    sig_fft = fftpack.fft(adc_data, axis=0)
    amplitude = np.abs(sig_fft)
    series = pd.Series(amplitude.flatten(), name='magnitude')
    df = pd.DataFrame(series)
    
    # Save the DataFrame to a new CSV file
    df.to_csv(os.path.join(directory, output_filename), index=False)


def main():
  for filename in os.listdir(present_dir):
    process_data(filename, present_dir, f"fft_magnitude_{filename}")

  for filename in os.listdir(absent_dir):
    process_data(filename, absent_dir, f"fft_magnitude_{filename}")


if __name__ == "__main__":
  main()