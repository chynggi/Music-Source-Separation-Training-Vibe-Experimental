"""BandSplit Subband ResUNet integration.

This module exposes the ByteDance-inspired subband ResUNet separator
for convenient import across the project.
"""

from .model import BandSplitSubbandResUNet

__all__ = ["BandSplitSubbandResUNet"]
