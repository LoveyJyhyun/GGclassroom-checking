from checker import aidetect


def test_short_text_not_judged():
    assert aidetect.heuristics("Too short.")["score"] is None


def test_ai_style_scores_higher_than_casual():
    ai = ("In conclusion, it is important to note that photosynthesis plays a crucial role in the realm of biology. "
          "Furthermore, plants harness sunlight to produce glucose and oxygen efficiently. "
          "Moreover, this process underscores the delicate balance of our planet's ecosystems. "
          "Additionally, chlorophyll absorbs light energy and converts it into chemical energy. "
          "Ultimately, photosynthesis is a testament to the complexity of natural systems. ") * 2
    casual = ("so i did the experiment with the plant and honestly it was kind of weird. "
              "the leaf in the dark went pale after like three days?? "
              "my sister said that's because no light means no sugar, which makes sense i guess. "
              "we measured stuff. "
              "the one by the window grew way more, almost 4cm more than the other one, which surprised me a lot because i thought water mattered more than light. ") * 2
    assert aidetect.heuristics(ai)["score"] > aidetect.heuristics(casual)["score"]
