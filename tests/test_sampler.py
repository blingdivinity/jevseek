"""The pure parts: candidate parsing and un-tempering, whitespace merging, the
plausibility floor, orderings, combination, order-bias stats, the repeat
penalty, sentence trimming, and preset merging. No network."""

import math
import random

import pytest

from jevseek import deepseek
from jevseek.deepseek import END, Candidate
from jevseek.sampler import (
    PRESETS, Config, combine, floor_candidates, jev_weight, merge_variants,
    order_stats, orderings, parse_orders, trim_to_boundary, _penalty,
)


def test_candidates_decode_bytes_and_drop_fragments():
    entries = [
        {"token": " Paris", "logprob": -0.1, "bytes": list(b" Paris")},
        {"token": "\\xe2\\x80", "logprob": -3.0, "bytes": [0xE2, 0x80]},   # half a multibyte char
        {"token": "<｜｜end▁of▁sentence｜｜>", "logprob": -2.0, "bytes": None},
        {"token": "", "logprob": -1.0, "bytes": []},
    ]
    cands = deepseek._candidates(entries, top_k=10)
    assert [c.text for c in cands] == [" Paris", END]


def test_candidates_merge_duplicate_spellings():
    entries = [{"token": " ", "logprob": math.log(0.3)}, {"token": " ", "logprob": math.log(0.2)}]
    cands = deepseek._candidates(entries, top_k=10)
    assert len(cands) == 1 and cands[0].p == pytest.approx(0.5)


def test_untempering_recovers_the_distribution():
    # The API reports log_softmax(logits / T). At T=2 every logprob is halved
    # (up to normalisation); multiplying back by T and renormalising must
    # return the T=1 distribution.
    true = {"a": 0.6, "b": 0.3, "c": 0.1}
    logits = {t: math.log(p) for t, p in true.items()}
    T = 2.0
    z = math.log(sum(math.exp(v / T) for v in logits.values()))
    reported = [{"token": t, "logprob": v / T - z} for t, v in logits.items()]
    cands = deepseek._candidates(reported, top_k=10, temperature=T)
    got = {c.text: c.p for c in cands}
    for t in true:
        assert got[t] == pytest.approx(true[t], abs=1e-9)


def test_merge_variants_folds_whitespace_and_keeps_likelier_surface():
    cands = [Candidate(" capital", math.log(0.9)), Candidate("capital", math.log(0.001)),
             Candidate(".", math.log(0.05)), Candidate(".\n\n", math.log(0.02)), Candidate("\n\n", math.log(0.01))]
    grouped, surface = merge_variants(cands, "space")
    labels = [c.text for c in grouped]
    assert labels == ["capital", ".", "\n\n"]
    assert surface["capital"] == " capital"      # deepseek's likelier spelling is emitted
    assert surface["."] == "."
    assert grouped[0].p == pytest.approx(0.901)


def test_merge_variants_case_level():
    cands = [Candidate(" Sea", math.log(0.2)), Candidate(" sea", math.log(0.7))]
    grouped, surface = merge_variants(cands, "case")
    assert [c.text for c in grouped] == ["sea"] and surface["sea"] == " sea"


def test_floor_keeps_end_and_drops_the_implausible():
    cands = [Candidate("a", math.log(0.9)), Candidate("b", math.log(0.0005)), Candidate(END, math.log(1e-6))]
    kept = floor_candidates(cands, 0.001)
    assert [c.text for c in kept] == ["a", END]
    assert floor_candidates(cands, 0.0) == cands


def test_orderings_shapes():
    cands = [Candidate("x", -0.1), Candidate("y", -1.0), Candidate("z", -2.0)]
    ords = orderings(cands, parse_orders("deepseek,reverse,alpha,random:2"), random.Random(0))
    assert ords["deepseek"] == ["x", "y", "z"]
    assert ords["reverse"] == ["z", "y", "x"]
    assert ords["alpha"] == ["x", "y", "z"]
    assert set(ords) == {"deepseek", "reverse", "alpha", "random0", "random1"}
    assert sorted(ords["random0"]) == ["x", "y", "z"]


def test_parse_orders_rejects_unknown():
    with pytest.raises(ValueError):
        parse_orders("upsidedown")


def test_combine_mean_and_stats():
    per = {"o1": {"a": 0.8, "b": 0.2}, "o2": {"a": 0.4, "b": 0.6}}
    m = combine(per, ["a", "b"], "mean")
    assert m["a"] == pytest.approx(0.6)
    st = order_stats(per, {"o1": ["a", "b"], "o2": ["b", "a"]}, m)
    assert st["agree"] == 0.5 and st["first_wins"] == 1.0 and st["distinct_winners"] == 2
    assert st["tv"] == pytest.approx(0.2)


def test_jev_weight_gates():
    sure = [Candidate("a", math.log(0.99)), Candidate("b", math.log(0.01))]
    torn = [Candidate("a", math.log(0.5)), Candidate("b", math.log(0.5))]
    assert jev_weight(sure, "none") == 1.0
    assert jev_weight(sure, "top1") == pytest.approx(0.01)
    assert jev_weight(torn, "top1") == pytest.approx(0.5)
    assert jev_weight(torn, "entropy") == pytest.approx(1.0)
    assert jev_weight(sure, "entropy") < 0.1


def test_repeat_penalty_matches_surface_forms():
    cfg = Config(repeat_penalty=1.6, repeat_window=16)
    tokens = [" yes", " yes", "Yes"]
    assert _penalty(tokens, "yes", cfg) == pytest.approx(1.6 ** 3)   # label vs surfaces, case-folded
    assert _penalty(tokens, " no", cfg) == 1.0
    assert _penalty(tokens, "\n\n", cfg) == 1.0                       # whitespace never penalised


def test_trim_to_boundary():
    assert trim_to_boundary("One. Two? Three lol anyway") == "One. Two?"
    assert trim_to_boundary("no boundary here") == "no boundary here"
    assert trim_to_boundary("Line one\nhalf a line") == "Line one"


def test_presets_are_valid_configs():
    for name, over in PRESETS.items():
        cfg = Config(**over)
        assert cfg.mode in ("chat", "raw", "base"), name
    # the essay preset tolerates a spike: rambling must hold for a window
    essay = Config(**PRESETS["essay"])
    assert essay.ramble_patience > 1 and essay.ramble_noul >= 0.8
    assert Config(**PRESETS["essay"]).gate == "top1"
    assert Config(**PRESETS["pure"]).stop_noul == 0.0


def test_ngram_block_stops_sentence_loops():
    cfg = Config(repeat_penalty=1.0, no_repeat_ngram=4)
    toks = ["The", " proof", " is", " not", " understandable", ".", " The", " proof", " is"]
    assert _penalty(toks, " not", cfg) >= 1e6       # completes "the proof is not" again
    assert _penalty(toks, " opaque", cfg) == 1.0    # a new 4-gram is fine
    assert _penalty(toks[:2], " is", cfg) == 1.0    # too short to have a prior n-gram
