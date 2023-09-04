import streamlit as st
import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder

import speech_recognition as sr


@st.cache(show_spinner=False)
def load_models():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    args = parser.parse_args()
    # Load the models and return them
    ensure_default_models(Path("saved_models"))
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_fpath)
    vocoder.load_model(args.voc_model_fpath)
    return encoder, synthesizer, vocoder


def main():
    # Set up the Streamlit app
    st.title("Voice Transfer")
    st.write("This is an example of a voice transfer application using Streamlit.")
    st.write("Please provide the necessary inputs and click the 'Generate' button.")

    # Load the models
    encoder, synthesizer, vocoder = load_models()

    # Input file selection
    st.subheader("Reference Voice")
    in_fpath = st.file_uploader("Upload an audio file (mp3, wav, m4a, flac)", type=["mp3", "wav", "m4a", "flac"])

    if in_fpath is not None:
        st.write("Reference voice selected.")

        # Preprocess the input audio
        preprocessed_wav = encoder.preprocess_wav(in_fpath)

        # Embed the utterance
        embed = encoder.embed_utterance(preprocessed_wav)

        # Text input
        st.subheader("Text Input")
        text = st.text_input("Enter a sentence to be synthesized (+-20 words)")

        # Generate button
        if st.button("Generate"):
            # Synthesize the spectrogram
            texts = [text]
            embeds = [embed]
            specs = synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]

            # Generate the waveform
            generated_wav = vocoder.infer_waveform(spec)

            # Play the audio
            st.audio(generated_wav, format="audio/wav")

            # Save the output
            st.subheader("Output")
            output_filename = "output.wav"
            sf.write(output_filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
            st.download_button("Download Output", data=open(output_filename, "rb"), file_name=output_filename)

if __name__ == "__main__":
    main()
