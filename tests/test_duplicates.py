from checker import duplicates as dp

ESSAY = ("The water cycle describes how water evaporates from the surface of the earth, "
         "rises into the atmosphere, cools and condenses into clouds, and falls again as rain.")


def test_copied_text_flagged():
    texts = {"a": ESSAY, "b": ESSAY + " I like rain.", "c": "My summer holiday was at the beach with my family."}
    r = dp.find(texts, use_embeddings=False, llm_review=0)
    assert [(p["a"], p["b"]) for p in r["pairs"]] == [("a", "b")]


def test_renamed_python_code_flagged():
    a = "def total(xs):\n    s = 0\n    for x in xs:\n        s += x  # add\n    return s\nprint(total([1,2,3]))\n"
    b = "def add_all(numbers):\n    acc = 0\n    for n in numbers:\n        acc += n\n    return acc\nprint(add_all([4,5,6]))\n"
    c = "import sys\nprint(sum(map(int, sys.stdin.read().split())))\n"
    r = dp.find({}, {"a": ("python", a), "b": ("python", b), "c": ("python", c)},
                use_embeddings=False, llm_review=0)
    assert [(p["a"], p["b"]) for p in r["pairs"]] == [("a", "b")]
    assert r["pairs"][0]["code"] == 1.0
