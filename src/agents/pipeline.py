"""
End-to-End Pipeline (Phase 3)
Video -> Segmentation -> Landmark Detection -> Sign Classification -> Gloss Translation -> English Text
"""

from src.preprocessing.video_segmenter import VideoSegmenter, VideoSegment


class ASLPipeline:
    """Full ASL video-to-text pipeline."""

    def __init__(self, alphabet_model_path=None, word_model_path=None, segmenter=None):
        self.segmenter = segmenter or VideoSegmenter()
        self.alphabet_model_path = alphabet_model_path
        self.word_model_path = word_model_path

    def predict_letter(self, image):
        """Phase 1: Single image -> letter."""
        raise NotImplementedError("Phase 1 - load MLP model and classify")

    def predict_word(self, video_clip):
        """Phase 2: Video clip -> word."""
        raise NotImplementedError("Phase 2 - load LSTM model and classify")

    def predict_sentence(self, video_path: str) -> str:
        """Phase 3: Full video -> English sentence.

        Steps:
            1. Segment video into individual sign clips
            2. Classify each clip into a gloss (word)
            3. Translate gloss sequence to English via LLM
        """
        # Step 1: Segment
        segments = self.segmenter.segment(video_path)

        # Step 2: Classify each segment -> gloss
        glosses = []
        for seg in segments:
            # TODO: classify segment using word model
            # gloss = self.predict_word(seg)
            # glosses.append(gloss)
            pass

        # Step 3: Translate gloss sequence to English
        # from src.agents.gloss_translator import translate_glosses
        # return translate_glosses(glosses)

        raise NotImplementedError(
            f"Segmentation works ({len(segments)} segments found). "
            "Classification and translation not yet implemented."
        )
