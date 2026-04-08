"""
End-to-End Pipeline (Phase 3)
Video -> Frame Extraction -> Landmark Detection -> Sign Classification -> Gloss Translation -> English Text
"""


class ASLPipeline:
    """Full ASL video-to-text pipeline."""

    def __init__(self, alphabet_model_path=None, word_model_path=None):
        raise NotImplementedError("Phase 3 - implement end-to-end pipeline")

    def predict_letter(self, image):
        """Phase 1: Single image -> letter."""
        raise NotImplementedError

    def predict_word(self, video_clip):
        """Phase 2: Video clip -> word."""
        raise NotImplementedError

    def predict_sentence(self, video):
        """Phase 3: Full video -> English sentence."""
        raise NotImplementedError
