"""Character threshold CLI tests that do not require the compiled Rust extension."""

import argparse
import unittest

from sglang_router.router_args import RouterArgs


class TestCacheMinMatchChars(unittest.TestCase):
    def test_default(self):
        parser = argparse.ArgumentParser()
        RouterArgs.add_cli_args(parser)
        args = RouterArgs.from_cli_args(parser.parse_args([]))
        self.assertEqual(args.cache_min_match_chars, 0)

    def test_plain_and_prefixed_options(self):
        for prefixed in (False, True):
            with self.subTest(prefixed=prefixed):
                parser = argparse.ArgumentParser()
                RouterArgs.add_cli_args(parser, use_router_prefix=prefixed)
                prefix = "router-" if prefixed else ""
                args = RouterArgs.from_cli_args(
                    parser.parse_args(
                        [f"--{prefix}cache-min-match-chars", "4096"]
                    ),
                    use_router_prefix=prefixed,
                )
                self.assertEqual(args.cache_min_match_chars, 4096)
                args._validate_router_args()

    def test_negative_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            RouterArgs(cache_min_match_chars=-1)._validate_router_args()


if __name__ == "__main__":
    unittest.main()
