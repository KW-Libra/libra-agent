from __future__ import annotations

import os
import unittest

from libra_agent.libra_api import _safe_exception_detail


class LibraApiErrorTests(unittest.TestCase):
    def test_safe_exception_detail_redacts_configured_keys(self) -> None:
        old_key = os.environ.get("GEMINI_API_KEY")
        os.environ["GEMINI_API_KEY"] = "secret-key"
        try:
            root = RuntimeError("request failed with key=secret-key")
            wrapped = ValueError("judge failed")
            wrapped.__cause__ = root

            detail = _safe_exception_detail(wrapped)

            self.assertIn("ValueError: judge failed", detail)
            self.assertIn("RuntimeError: request failed with key=<redacted>", detail)
            self.assertNotIn("secret-key", detail)
        finally:
            if old_key is None:
                os.environ.pop("GEMINI_API_KEY", None)
            else:
                os.environ["GEMINI_API_KEY"] = old_key


if __name__ == "__main__":
    unittest.main()
