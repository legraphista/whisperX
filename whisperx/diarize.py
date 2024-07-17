import numpy as np
import pandas as pd
from pyannote.audio import Pipeline
from typing import Optional, Union
import torch

from .audio import load_audio, SAMPLE_RATE


class DiarizationPipeline:
    def __init__(
        self,
        model_name="pyannote/speaker-diarization-3.1",
        use_auth_token=None,
        device: Optional[Union[str, torch.device]] = "cpu",
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.model = Pipeline.from_pretrained(model_name, use_auth_token=use_auth_token).to(device)

    def __call__(self, audio: Union[str, np.ndarray], num_speakers=None, min_speakers=None, max_speakers=None, return_embeddings=False):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE
        }

        result = self.model(audio_data, num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers, return_embeddings=return_embeddings)

        diarization = result[0] if return_embeddings else result
        embeddings = result[1] if return_embeddings else None

        diarize_df = pd.DataFrame(diarization.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)

        speaker_embeddings = {speaker: embeddings[s].tolist() for s, speaker in enumerate(diarization.labels())} if embeddings is not None else None

        return diarize_df, speaker_embeddings


def assign_speaker(diarize_df, start, end, fill_nearest=False):
    intersections = np.minimum(diarize_df['end'], end) - np.maximum(diarize_df['start'], start)
    valid_segments = diarize_df if fill_nearest else diarize_df[intersections > 0]

    if not valid_segments.empty:
        return valid_segments.loc[intersections[valid_segments.index].idxmax(), 'speaker']
    return None


def assign_word_speakers(diarize_df, transcript_result, speaker_embeddings=None, fill_nearest=False):
    new_segments = []

    for seg in transcript_result["segments"]:
        if 'words' not in seg or not seg['words']:
            speaker = assign_speaker(diarize_df, seg['start'], seg['end'], fill_nearest)
            new_segments.append({**seg, "speaker": speaker})
        else:
            current_segment = {
                "start": seg["start"],
                "end": None,
                "text": "",
                "words": [],
                "speaker": None
            }

            for word in seg['words']:
                word_start = word.get('start', seg['start'])
                word_end = word.get('end', seg['end'])
                speaker = assign_speaker(diarize_df, word_start, word_end, fill_nearest)
                word["speaker"] = speaker

                if current_segment["speaker"] != speaker:
                    if current_segment["words"]:
                        current_segment["end"] = word_start
                        current_segment["text"] = " ".join(w["word"] for w in current_segment["words"])
                        new_segments.append(current_segment)

                    current_segment = {
                        "start": word_start,
                        "end": None,
                        "text": "",
                        "words": [],
                        "speaker": speaker
                    }

                current_segment["words"].append(word)

            if current_segment["words"]:
                current_segment["end"] = seg["end"]
                current_segment["text"] = " ".join(w["word"] for w in current_segment["words"])
                new_segments.append(current_segment)

    transcript_result["segments"] = new_segments

    if speaker_embeddings is not None:
        transcript_result["speaker_embeddings"] = speaker_embeddings

    return transcript_result


class Segment:
    def __init__(self, start, end, speaker=None):
        self.start = start
        self.end = end
        self.speaker = speaker
