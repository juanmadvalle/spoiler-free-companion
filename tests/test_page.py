"""The web page keeps its promises: every text exists in both languages, the guide is wired, no inline styles."""
import re
import unittest
from pathlib import Path

PAGE = Path(__file__).resolve().parent.parent / "companion" / "static" / "index.html"


class PageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = PAGE.read_text(encoding="utf-8")
        script = cls.html[cls.html.index("<script"):]
        cls.texts = {}
        for language in ("es", "en"):
            block = script[script.index(f"  {language}: {{"):]
            block = block[:block.index("\n  },")]
            # a key starts a line or follows a comma or brace; "Terminado: " inside a text is not a key
            cls.texts[language] = set(re.findall(r'(?:^|[,{])\s*([A-Za-z][A-Za-z0-9]*):\s*"', block, re.M))
        cls.script = script

    def test_every_text_used_by_the_page_exists_in_spanish_and_english(self):
        used = set(re.findall(r'data-i18n(?:-ph)?="([^"]+)"', self.html))
        used |= set(re.findall(r'\bt\("([A-Za-z0-9]+)"\)', self.script))
        used |= {step + part for step in ("g1", "g1b", "g2", "g2v", "g3", "g4") for part in ("Title", "Text")}  # built at run time
        for language, available in self.texts.items():
            self.assertEqual(sorted(used - available), [], f"missing in {language}")
        self.assertEqual(self.texts["es"], self.texts["en"], "the two languages must define the same keys")

    def test_the_guide_has_its_pieces_and_the_boxes_it_highlights(self):
        for element in ('id="guide"', 'id="guideTitle"', 'id="guideText"', 'id="steps"', 'id="examples"', 'id="llmNote"',
                        'id="helpDialog"', 'id="openHelp"', 'id="libCard"', 'id="addCard"', 'id="posCard"', 'id="askCard"'):
            self.assertIn(element, self.html)
        self.assertIn('aria-live="polite"', self.html)  # screen readers hear the next step
        for call in ("updateGuide();", "loadExamples();"):
            self.assertIn(call, self.script)

    def test_the_content_security_policy_still_works_with_the_page(self):
        self.assertNotIn(' style="', self.html)  # inline style attributes would be blocked by the policy
        self.assertNotRegex(self.html, r"\son[a-z]+=")  # no inline event handlers
        self.assertEqual(self.html.count("__NONCE__"), 2)  # one for the style block, one for the script
        self.assertNotRegex(self.html, r"(src|href)=\"https?://")  # nothing loaded from the internet

    def test_the_language_switch_is_always_visible_and_remembers_the_choice(self):
        self.assertIn('class="lang"', self.html)
        self.assertIn('data-lang="es"', self.html)
        self.assertIn('data-lang="en"', self.html)
        self.assertRegex(self.html, r"\.lang \{ position: fixed;")  # stays on screen while the page scrolls
        self.assertIn('localStorage.setItem("lang"', self.script)
        self.assertIn('localStorage.getItem("lang")', self.script)
        self.assertIn("aria-pressed", self.script)  # the current language is announced to screen readers

    def test_the_brand_is_centred_and_named(self):
        self.assertIn('class="brand"', self.html)
        self.assertRegex(self.html, r"header\.top \{[^}]*text-align: center")
        # the same name in both languages, shown large in capitals, and no bookmark symbol
        self.assertEqual(self.html.count('title: "Spoiler-free companion"'), 2)
        self.assertRegex(self.html, r"\.brand h1 \{[^}]*text-transform: uppercase")
        self.assertNotIn("Marcapáginas", self.html)
        self.assertNotIn("Bookmark", self.html)
        self.assertNotIn('class="mark"', self.html)

    def test_no_text_is_left_unused_and_nothing_is_said_twice(self):
        code = self.script[: self.script.index("  es: {")] + self.script[self.script.index("\n  },", self.script.index("  en: {")):]
        dynamic = {step + part for step in ("g1", "g1b", "g2", "g2v", "g3", "g4") for part in ("Title", "Text")}
        for key in sorted(self.texts["es"]):
            used = (f'data-i18n="{key}"' in self.html or f'data-i18n-ph="{key}"' in self.html or f'"{key}"' in code
                    or key in dynamic)
            self.assertTrue(used, f"text '{key}' is defined but never shown")
        # the promise and the steps are stated once each: in the guide, not again in a welcome card
        for repeated in ('id="empty"', "welcomeTitle", "peopleHint"):
            self.assertNotIn(repeated, self.html)
        for language in ("es", "en"):  # the step stepper already shows the number, so titles do not say "Step N of 3"
            for key in ("g1Title", "g1bTitle", "g2Title", "g2vTitle", "g3Title"):
                self.assertNotRegex(self.text_value(language, key), r"(Paso|Step) \d")

    def text_value(self, language, key):
        block = self.script[self.script.index(f"  {language}: {{"):]
        return re.search(key + r':\s*"([^"]*)"', block).group(1)


if __name__ == "__main__":
    unittest.main()
