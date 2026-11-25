from hearxiv.narrate.spoken import apply_lexicon, latex_to_words, strip_citations, to_spoken, verbalize_numbers


def test_citations_removed():
    assert strip_citations("as shown before [cite:bib.bib3,bib.bib4].") == "as shown before."


def test_lexicon_whole_words_and_trailing_hyphen():
    assert apply_lexicon("LoRA and ReLU on arXiv") == "lora and ray-loo on archive"
    assert apply_lexicon("CIFAR-10") == "see-far-10"
    assert apply_lexicon("FLoRAx") == "FLoRAx"  # no partial-word replacement


def test_glossary_overrides_lexicon():
    assert apply_lexicon("GPT", {"GPT": "jee pee tee"}) == "jee pee tee"


def test_numbers():
    assert verbalize_numbers("lr 1e-4") == "lr 1 times ten to the minus 4"
    assert verbalize_numbers("a 7B model") == "a 7 billion model"
    assert verbalize_numbers("224x224 crops") == "224 by 224 crops"
    assert verbalize_numbers("2x faster") == "2 times faster"


def test_nested_latex():
    s = latex_to_words(r"\frac{QK^{T}}{\sqrt{d_{k}}}")
    assert s == "QK transpose over square root of d sub k"
    assert latex_to_words("O(n^2)") == "O of n squared"
    assert latex_to_words(r"\alpha") == "alpha"


def test_full_pass():
    out = to_spoken(r"We fine-tune a 7B LLM with $\alpha=0.5$ [cite:bib.bib1] (see Fig. 3).")
    assert out == "We fine-tune a 7 billion L L M with alpha equals 0.5 (see Figure 3)."
