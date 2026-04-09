"""Tests for EEE-specific preprocessing: metric extraction and benchmark name cleaning."""
from eval_entity_resolver.eee import clean_eval_name, extract_metric


class TestExtractMetric:
    # --- "X on Y" pattern ---

    def test_strips_on_suffix(self):
        assert extract_metric("Accuracy on IFEval") == "Accuracy"

    def test_strips_on_suffix_multiword(self):
        assert extract_metric("Exact Match on MATH Level 5") == "Exact Match"

    def test_strips_em_abbreviation(self):
        assert extract_metric("EM on GSM8K") == "EM"

    # --- Short metric names pass through unchanged ---

    def test_preserves_bare_metric(self):
        assert extract_metric("Accuracy") == "Accuracy"

    def test_preserves_empty(self):
        assert extract_metric("") == ""

    def test_preserves_short_name(self):
        assert extract_metric("F1") == "F1"

    def test_preserves_two_word_name(self):
        assert extract_metric("Win Rate") == "Win Rate"

    # --- Dot notation ---

    def test_dot_notation_extracts_accuracy(self):
        assert extract_metric("bfcl.live.live_accuracy") == "Accuracy"

    def test_dot_notation_extracts_ast_accuracy(self):
        assert extract_metric("bfcl.non_live.simple_ast_accuracy") == "AST Accuracy"

    def test_ast_accuracy_distinct_from_accuracy(self):
        """AST accuracy (function call AST matching) is a different metric from plain accuracy."""
        assert extract_metric("Non-live simple AST accuracy") == "AST Accuracy"
        assert extract_metric("Live accuracy") == "Accuracy"

    def test_dot_notation_extracts_win_rate(self):
        assert extract_metric("fibble1_arena.win_rate") == "Win Rate"

    def test_dot_notation_extracts_rank(self):
        assert extract_metric("bfcl.overall.rank") == "rank"

    def test_dot_notation_extracts_cost(self):
        assert extract_metric("bfcl.overall.total_cost_usd") == "cost"

    def test_dot_notation_extracts_latency(self):
        assert extract_metric("bfcl.overall.latency_mean_s") == "mean-latency"

    def test_dot_notation_extracts_stddev(self):
        assert extract_metric("bfcl.format_sensitivity.stddev") == "stddev"

    # --- Verbose descriptions → keyword extraction ---

    def test_description_extracts_keyword(self):
        assert extract_metric("Chat accuracy - includes easy chat subsets") == "Accuracy"

    def test_description_extracts_multiword_keyword(self):
        assert extract_metric("Corporate lawyer world mean score.") == "Mean Score"

    def test_no_keyword_description_falls_back_to_score(self):
        assert extract_metric("Global MMLU Lite - Arabic") == "score"

    def test_first_keyword_wins_by_position(self):
        # "score" appears before "accuracy" in this description
        assert extract_metric("Factuality score - measures factual accuracy") == "score"


class TestCleanEvalName:
    # --- Dot notation (last segment is metric, rest is benchmark) ---

    def test_dot_drops_last_segment(self):
        assert clean_eval_name("bfcl.overall.rank") == "bfcl overall"

    def test_dot_two_segment_benchmark(self):
        assert clean_eval_name("bfcl.overall.overall_accuracy") == "bfcl overall"

    def test_dot_live_category(self):
        assert clean_eval_name("bfcl.live.live_accuracy") == "bfcl live"

    def test_dot_non_live_category(self):
        assert clean_eval_name("bfcl.non_live.simple_ast_accuracy") == "bfcl non live"

    # --- Underscore metric suffix (fibble/wordle) ---

    def test_underscore_strips_win_rate(self):
        assert clean_eval_name("fibble1_arena_win_rate") == "fibble1 arena"

    def test_underscore_strips_avg_attempts(self):
        assert clean_eval_name("wordle_arena_avg_attempts") == "wordle arena"

    def test_underscore_strips_elo(self):
        assert clean_eval_name("overall_elo") == "overall"

    # --- Trailing metric words (ACE/APEX) ---

    def test_trailing_score(self):
        assert clean_eval_name("Gaming Score") == "Gaming"

    def test_trailing_pass_at_1(self):
        assert clean_eval_name("Investment Banking Pass@1") == "Investment Banking"

    def test_trailing_mean_score(self):
        assert clean_eval_name("Corporate Lawyer Mean Score") == "Corporate Lawyer"

    # --- Clean names pass through ---

    def test_clean_name_unchanged(self):
        assert clean_eval_name("IFEval") == "IFEval"

    def test_multi_word_clean_unchanged(self):
        assert clean_eval_name("Chat Hard") == "Chat Hard"

    def test_empty(self):
        assert clean_eval_name("") == ""
