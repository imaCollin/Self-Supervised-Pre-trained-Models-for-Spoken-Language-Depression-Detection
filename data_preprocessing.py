import os
import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

def load_audio(file_path, target_sr=16000):
    """
    加载音频文件并重采样到目标采样率。

    Args:
        file_path (str): 音频文件路径。
        target_sr (int): 目标采样率，默认为16000。

    Returns:
        tuple: (信号数据, 采样率)
    """
    try:
        signal, sr = librosa.load(file_path, sr=target_sr)
        return signal, sr
    except Exception as e:
        raise ValueError(f"无法加载音频文件 {file_path}，错误信息: {e}")

  def extract_mfcc(signal, sr, n_mfcc=40):
    """
    提取 MFCC 特征。

    Args:
        signal (numpy.ndarray): 音频信号。
        sr (int): 采样率。
        n_mfcc (int): 要提取的MFCC特征数量，默认40。

    Returns:
        numpy.ndarray: MFCC 特征。
    """
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def extract_mel_spectrogram(signal, sr, n_mels=128):
    """
    提取 Mel-Spectrogram 特征。

    Args:
        signal (numpy.ndarray): 音频信号。
        sr (int): 采样率。
        n_mels (int): Mel频谱图的频带数量，默认128。

    Returns:
        numpy.ndarray: Mel-Spectrogram 特征。
    """
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # 转换为对数刻度
    return mel_spec_db

def extract_wav2vec2_features(signal, sr, model_name="facebook/wav2vec2-base-960h"):
    """
    使用 wav2vec2 提取高维特征。

    Args:
        signal (numpy.ndarray): 音频信号。
        sr (int): 采样率。
        model_name (str): wav2vec2 模型名称，默认 "facebook/wav2vec2-base-960h"。

    Returns:
        torch.Tensor: wav2vec2 特征。
    """
    # 加载模型和处理器
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)

    # 转换为模型输入格式
    input_values = processor(signal, sampling_rate=sr, return_tensors="pt", padding=True).input_values

    # 提取特征
    with torch.no_grad():
        embeddings = model(input_values).last_hidden_state
    return embeddings

def add_noise(signal, noise_level=0.005):
    """
    增加随机噪声。

    Args:
        signal (numpy.ndarray): 音频信号。
        noise_level (float): 噪声强度，默认0.005。

    Returns:
        numpy.ndarray: 增强后的音频信号。
    """
    noise = noise_level * np.random.randn(len(signal))
    return signal + noise

def time_stretch(signal, rate=1.1):
    """
    改变音频的时间长度（时间缩放）。

    Args:
        signal (numpy.ndarray): 音频信号。
        rate (float): 时间缩放比例，默认为1.1。

    Returns:
        numpy.ndarray: 时间缩放后的音频信号。
    """
    return librosa.effects.time_stretch(signal, rate)

def pitch_shift(signal, sr, n_steps=2):
    """
    改变音频的音调。

    Args:
        signal (numpy.ndarray): 音频信号。
        sr (int): 采样率。
        n_steps (int): 音调变化的半音数，默认为2。

    Returns:
        numpy.ndarray: 调整音调后的音频信号。
    """
    return librosa.effects.pitch_shift(signal, sr, n_steps)

def process_audio_file(file_path, feature_type="mfcc", target_sr=16000, model_name=None):
    """
    对单个音频文件进行特征提取。

    Args:
        file_path (str): 音频文件路径。
        feature_type (str): 特征类型（"mfcc", "mel", "wav2vec2"）。
        target_sr (int): 目标采样率。
        model_name (str): wav2vec2 模型名称（仅当 feature_type="wav2vec2" 时需要）。

    Returns:
        numpy.ndarray 或 torch.Tensor: 提取的特征。
    """
    signal, sr = load_audio(file_path, target_sr)

    if feature_type == "mfcc":
        return extract_mfcc(signal, sr)
    elif feature_type == "mel":
        return extract_mel_spectrogram(signal, sr)
    elif feature_type == "wav2vec2":
        if model_name is None:
            raise ValueError("使用 wav2vec2 提取特征时必须提供 model_name")
        return extract_wav2vec2_features(signal, sr, model_name)
    else:
        raise ValueError(f"不支持的特征类型：{feature_type}")

def batch_process_audio_files(file_paths, feature_type="mfcc", target_sr=16000, model_name=None):
    """
    批量处理音频文件并提取特征。

    Args:
        file_paths (list): 音频文件路径列表。
        feature_type (str): 特征类型（"mfcc", "mel", "wav2vec2"）。
        target_sr (int): 目标采样率。
        model_name (str): wav2vec2 模型名称（仅当 feature_type="wav2vec2" 时需要）。

    Returns:
        list: 提取的特征列表。
    """
    features = []
    for file_path in file_paths:
        try:
            feature = process_audio_file(file_path, feature_type, target_sr, model_name)
            features.append(feature)
        except Exception as e:
            print(f"处理文件 {file_path} 出错：{e}")
    return features
