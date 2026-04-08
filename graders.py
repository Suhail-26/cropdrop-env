# graders.py (root level)
import sys
import os

_server_path = os.path.join(os.path.dirname(__file__), "server")
if _server_path not in sys.path:
    sys.path.insert(0, _server_path)

try:
    from server.graders import easy_grader, medium_grader, hard_grader
except ImportError:
    from graders import easy_grader, medium_grader, hard_grader  # fallback

__all__ = ["easy_grader", "medium_grader", "hard_grader"]
