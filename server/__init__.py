# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cropdrop Env environment server components."""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .cropdrop_env_environment import CropdropEnvironment

try:
    from graders import EasyGrader, MediumGrader, HardGrader
except ImportError:
    # Fallback if graders are not available yet
    EasyGrader = None
    MediumGrader = None
    HardGrader = None

__all__ = [
    "CropdropEnvironment",
    "EasyGrader",
    "MediumGrader",
    "HardGrader",
]
