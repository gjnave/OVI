import os
import tempfile
from typing import Optional

import numpy as np
from moviepy.editor import ImageSequenceClip, AudioFileClip
from scipy.io import wavfile


def save_video(
    output_path: str,
    video_numpy: np.ndarray,
    audio_numpy: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
    fps: int = 24,
) -> str:
    """
    Combine a sequence of video frames with an optional audio track and save as an MP4.

    Args:
        output_path (str): Path to the output MP4 file.
        video_numpy (np.ndarray): Numpy array of frames. Shape (C, F, H, W).
                                  Values can be in range [-1, 1] or [0, 255].
        audio_numpy (Optional[np.ndarray]): 1D or 2D numpy array of audio samples, range [-1, 1].
        sample_rate (int): Sample rate of the audio in Hz. Defaults to 16000.
        fps (int): Frames per second for the video. Defaults to 24.

    Returns:
        str: Path to the saved MP4 file.
    """

    # --- Validate inputs ---
    assert isinstance(video_numpy, np.ndarray), "video_numpy must be a numpy array"
    assert video_numpy.ndim == 4, "video_numpy must have shape (C, F, H, W)"
    assert video_numpy.shape[0] in {1, 3}, "video_numpy must have 1 or 3 channels"

    if audio_numpy is not None:
        assert isinstance(audio_numpy, np.ndarray), "audio_numpy must be a numpy array"
        assert np.abs(audio_numpy).max() <= 1.0, "audio_numpy values must be in range [-1, 1]"

    # --- Normalize & reformat frames ---
    video_numpy = video_numpy.transpose(1, 2, 3, 0)
    if video_numpy.max() <= 1.0:
        video_numpy = np.clip(video_numpy, -1, 1)
        video_numpy = ((video_numpy + 1) / 2 * 255).astype(np.uint8)
    else:
        video_numpy = video_numpy.astype(np.uint8)

    frames = list(video_numpy)
    clip = ImageSequenceClip(frames, fps=fps)

    wav_path = None
    audio_clip = None
    final_clip = clip

    try:
        # --- Write temp WAV safely ---
        if audio_numpy is not None:
            fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)  # release file handle (important on Windows)
            wavfile.write(
                wav_path,
                sample_rate,
                (audio_numpy * 32767).astype(np.int16),
            )
            audio_clip = AudioFileClip(wav_path)
            final_clip = clip.set_audio(audio_clip)

        # --- Write final MP4 ---
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=fps,
            verbose=False,
            logger=None,
        )

    finally:
        # --- Cleanup ---
        try:
            if audio_clip is not None:
                audio_clip.close()
        except Exception:
            pass
        try:
            if final_clip is not None and final_clip is not clip:
                final_clip.close()
        except Exception:
            pass
        try:
            clip.close()
        except Exception:
            pass
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass

    return output_path
