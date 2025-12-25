import os
import torch
from pydub import AudioSegment
from faster_whisper import WhisperModel
import io


class Transcribe:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.model = WhisperModel("medium", device=device, compute_type="int8")

    def transcribe(self, audio_path):
        result_audio = AudioSegment.silent(duration=0)
        result_text = ''

        audio = AudioSegment.from_file(audio_path)
        max_len = len(audio)

        segments, info = self.model.transcribe(audio_path, beam_size=5, word_timestamps=True)
        segments = list(segments)
        if not segments:
            raise ValueError(f'transcribe failed: {audio_path}')
        for i, seg in enumerate(segments):
            # process with the time
            if i == 0:
                start_time = max(0, seg.start)

            end_time = seg.end

            # calculate confidence
            if len(seg.words) > 0:
                confidence = sum([s.probability for s in seg.words]) / len(seg.words)
            else:
                confidence = 0.

            # clean text
            text = seg.text.replace('...', '')

            # left 0.08s for each audios
            audio_seg = audio[int(start_time * 1000): min(max_len, int(end_time * 1000) + 80)]

            # save = audio_seg.duration_seconds > 0.3 and \
            #         len(text) >= 1 and len(text) < 200 and \
            #         confidence > 0.5 \
            save = len(text) >= 1
            if save:
                result_audio += audio_seg
                result_text += text

            if i < len(segments) - 1:
                start_time = max(0, segments[i+1].start - 0.08)

        if not result_audio or not result_text:
            raise ValueError(f'transcribe failed: {audio_path}')
        vf = io.BytesIO()
        result_audio.export(vf, format='wav')
        vf.seek(0)
        return vf, result_text

