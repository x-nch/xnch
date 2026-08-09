class VoiceError(Exception):
    """Base class for voice pipeline errors."""


class VoiceSubsystemUnavailable(VoiceError):
    """STT/TTS backend or models are not installed or failed to load."""
