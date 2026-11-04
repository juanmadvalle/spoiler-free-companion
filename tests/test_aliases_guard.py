"""Alias expansion is position-aware; questions about the future are refused without any search or model call."""
import unittest

from companion import answer
from companion.agent import Session, run_agent
from companion.guard import REFUSAL, is_future_question
from companion.ingest import Chapter
from companion.store import Library


def story(*sentences):
    return Chapter("c", [s for s in sentences])


FILLER = "walked slowly along the grey harbour wall while the evening tide came in"
CH1 = story(*[f"Yesterday we saw Anna Reed {FILLER}, and Anna Reed said nothing {i}." for i in range(3)])
CH2 = story(*[f"The old pier belonged to Reed alone and nobody else {FILLER} {i}." for i in range(3)])
CH3 = story(*[f"Later we met Mark Reed {FILLER}, and Mark Reed smiled {i}." for i in range(3)])


class AliasTests(unittest.TestCase):
    def setUp(self):
        self.library = Library(":memory:")
        self.library.add_work("S", [CH1, CH2, CH3])
        self.addCleanup(self.library.close)
        self.h = [self.library.horizon_after_chapter("S", n) for n in (1, 2, 3)]

    def test_first_name_finds_passages_that_only_use_the_surname(self):
        found = self.library.search("S", "Anna", self.h[1], limit=20)
        self.assertTrue(any("Anna" not in p.text and "Reed" in p.text for p in found))

    def test_expansion_is_position_aware_and_drops_when_ambiguous(self):
        self.assertEqual(self.library.aliases("S", ["reed"], self.h[1]), ["anna"])
        # After Mark Reed is introduced, "reed" is ambiguous: no guessing, and never a future name before.
        self.assertEqual(self.library.aliases("S", ["reed"], self.h[2]), [])
        for horizon in self.h[:2]:
            self.assertNotIn("mark", self.library.aliases("S", ["reed", "anna"], horizon))

    def test_future_name_never_used_before_it_appears(self):
        found = self.library.search("S", "Reed", self.h[1], limit=50)
        self.assertTrue(found)
        self.assertTrue(all("Mark" not in p.text for p in found))


class GuardTests(unittest.TestCase):
    def test_future_questions_are_detected(self):
        for question in ("How does it end?", "How will the story end?", "What happens in the end?",
                         "What is going to happen to Holmes?", "Will she die?", "Tell me the ending",
                         "Give me a spoiler", "¿Cómo acaba el libro?", "¿Qué pasará con Ana?", "Cuéntame el final"):
            self.assertTrue(is_future_question(question), question)

    def test_questions_about_the_past_are_not_refused(self):
        for question in ("Who is Jabez Wilson?", "What happens when Holmes visits the house?",
                         "Who dies in this chapter?", "Why did she leave in the end of the letter?",
                         "What did he say later that night?", "¿Quién es Ana?", "¿Qué pasa en el capítulo anterior?"):
            self.assertFalse(is_future_question(question), question)

    def test_refused_before_searching_or_calling_a_model(self):
        library = Library(":memory:")
        self.addCleanup(library.close)
        library.add_work("S", [CH1])

        def forbidden(*args, **kwargs):
            raise AssertionError("must not be called")

        library.search = forbidden
        text, passages = answer.ask(library, "S", "How does it end?", 1, forbidden)
        self.assertEqual((text, passages), (REFUSAL, []))
        session = Session(library, "S", 1)
        self.assertEqual(run_agent(session, "How does it end?", client=forbidden), REFUSAL)


if __name__ == "__main__":
    unittest.main()
