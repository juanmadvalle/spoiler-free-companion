"""Character cards use only what the reader has reached: a name does not exist before its mentions do."""
import json
import unittest

from companion.agent import TOOLS, Session
from companion.cli import format_card
from companion.ingest import Chapter
from companion.store import Library

PAD = " The evening wind carried the smell of salt across the grey harbour while the lamplighter climbed his ladder."


def chapter(title, *sentences):
    return Chapter(title, [s + PAD for s in sentences])


BOOK = [
    chapter("One",
            "Captain Marlow arrived at the lighthouse. Marlow said nothing to the keeper Ines.",
            "Ines, the keeper of the light, was a quiet woman. Marlow asked Ines about the island.",
            "Then Ines replied that Marlow could stay. Marlow watched Ines light the lamp."),
    chapter("Two",
            "Marlow and Ines walked down to the cellar. Ines said the lantern was old.",
            "Marlow found the lantern. Ines answered that it had belonged to Captain Aldrich."),
    chapter("Three",
            "At last Zephyrine appeared at the door. Zephyrine, the harbour master, was the traitor.",
            "Zephyrine said that Marlow had been betrayed. Ines cried when Zephyrine spoke.",
            "Zephyrine laughed and Marlow understood that Zephyrine had planned everything.",
            "Ines looked at Zephyrine and Zephyrine looked away from Ines. Marlow waited by the door.",
            "Later Ines and Zephyrine argued about the lantern while Marlow listened in the cellar.",
            "Zephyrine told Ines that the lantern was a lie. Ines wept and Marlow left the island."),
]


class CardTests(unittest.TestCase):
    def setUp(self):
        self.library = Library(":memory:")
        self.library.add_work("B", BOOK)
        self.addCleanup(self.library.close)
        self.h = [self.library.horizon_after_chapter("B", n) for n in (1, 2, 3)]

    def names(self, horizon):
        return [c["name"] for c in self.library.characters("B", horizon)]

    def test_a_name_does_not_exist_before_its_first_mentions(self):
        self.assertNotIn("Zephyrine", self.names(self.h[0]))
        self.assertNotIn("Zephyrine", self.names(self.h[1]))
        self.assertIn("Zephyrine", self.names(self.h[2]))
        with self.assertRaises(LookupError):
            self.library.card("B", "Zephyrine", self.h[1])

    def test_cards_never_mention_later_names_or_facts(self):
        for horizon in self.h[:2]:
            blob = json.dumps([self.library.card("B", name, horizon) for name in self.names(horizon)])
            self.assertNotIn("Zephyrine", blob)
            self.assertNotIn("traitor", blob)
            self.assertNotIn("betrayed", blob)

    def test_card_content_grows_with_the_position(self):
        early = self.library.card("B", "Ines", self.h[0])
        late = self.library.card("B", "Ines", self.h[2])
        self.assertEqual(early["chapters"], [1])
        self.assertEqual(late["chapters"], [1, 2, 3])
        self.assertLess(early["mentions"], late["mentions"])
        self.assertEqual(early["kind"], "character")
        self.assertTrue(any("Zephyrine" == r["name"] for r in late["related"]))
        self.assertFalse(any("Zephyrine" == r["name"] for r in early["related"]))
        self.assertIn("Ines", early["first"]["text"])
        self.assertTrue(any("keeper" in d["text"] for d in early["described_as"]))

    def test_lookup_is_case_and_accent_insensitive(self):
        self.assertEqual(self.library.card("B", "ines", self.h[1])["name"], "Ines")
        self.assertEqual(self.library.card("B", "INES", self.h[1])["name"], "Ines")

    def test_agent_tool_has_no_position_parameter_and_cannot_see_ahead(self):
        tool = next(t for t in TOOLS if t["name"] == "character_card")
        self.assertEqual(set(tool["input_schema"]["properties"]), {"name"})
        session = Session(self.library, "B", self.h[0])
        self.assertIn("No card", session.run_tool("character_card", {"name": "Zephyrine", "horizon": 10**6}))
        self.assertNotIn("Zephyrine", session.run_tool("character_card", {"name": "Ines", "horizon": 10**6}))

    def test_cli_format_and_old_libraries_build_the_index_on_first_use(self):
        text = format_card(self.library.card("B", "Marlow", self.h[1]))
        self.assertIn("Marlow", text)
        self.assertIn("First seen", text)
        self.library.db.execute("DELETE FROM mentions")  # a library made before cards existed
        self.assertIn("Ines", self.names(self.h[1]))

    def test_reingest_replaces_the_mentions(self):
        self.library.add_work("B", BOOK[:1])
        top = self.library.horizon_after_chapter("B", 1)
        self.assertNotIn("Zephyrine", self.names(top))
        self.assertEqual(self.library.db.execute("SELECT COUNT(DISTINCT work_id) FROM mentions").fetchone()[0], 1)


class SpanishCardTests(unittest.TestCase):
    def test_spanish_names_with_accents_and_titles(self):
        pad = " La tarde caía sobre el puerto gris mientras el farolero subía despacio por su escalera."
        book = [Chapter("Capítulo I", [
                    "Doña Inés, la guardiana del faro, era una mujer callada." + pad,
                    "—¿Quién vive aquí? —preguntó Marlow. —Nadie —respondió Inés." + pad,
                    "Marlow miró a Inés y Inés dijo que podía quedarse." + pad]),
                Chapter("Capítulo II", ["Por fin apareció Céfiro, el traidor del puerto." + pad * 2,
                                        "Céfiro dijo que Céfiro lo sabía todo." + pad])]
        library = Library(":memory:")
        self.addCleanup(library.close)
        library.add_work("Faro", book)
        first = library.horizon_after_chapter("Faro", 1)
        names = [c["name"] for c in library.characters("Faro", first)]
        self.assertIn("Inés", names)
        self.assertNotIn("Céfiro", names)
        card = library.card("Faro", "ines", first)  # typed without the accent
        self.assertEqual(card["name"], "Inés")
        self.assertTrue(any("guardiana" in d["text"] for d in card["described_as"]))


if __name__ == "__main__":
    unittest.main()
