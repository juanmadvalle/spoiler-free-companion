"""Spanish support: stemming, language detection, names with accents, search, locate and spoiler safety."""
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from companion import answer, lang
from companion.cli import main
from companion.guard import REFUSAL_ES, is_future_question, nothing_found, refusal
from companion.ingest import Chapter, read_book
from companion.names import extract_names
from companion.store import Library

PAD = " La tarde caía sobre el puerto gris mientras el farolero subía despacio por su escalera de madera."
LIBRO = [
    Chapter("Capítulo primero", [
        "El capitán Marlow llegó al faro y conoció a la guardiana, doña Inés, que apenas hablaba." + PAD,
        "—¿Quién vive en la isla? —preguntó Marlow. —Nadie —respondió Inés mirando el mar." + PAD]),
    Chapter("Capítulo II", [
        "Inés le enseñó la bodega donde colgaba un farol oxidado junto a la puerta." + PAD * 2,
        "Los marineros corrían por el muelle porque la tormenta se acercaba deprisa." + PAD]),
    Chapter("Capítulo III", [
        "Por fin Inés confesó que la llave de ámbar estaba bajo el tercer escalón y que había traicionado "
        "al capitán." + PAD * 2]),
]


class StemmerTests(unittest.TestCase):
    def test_inflected_forms_share_a_stem(self):
        for group in (("corrió", "corriendo", "correr", "corre"), ("niño", "niños", "niña", "niñas"),
                      ("canción", "canciones"), ("casa", "casas"), ("caballero", "caballeros"),
                      ("marinero", "marineros"), ("habló", "hablaba", "hablando", "hablar")):
            self.assertEqual(len({lang.stem_es(w) for w in group}), 1, group)

    def test_unrelated_words_stay_apart_and_accents_are_folded(self):
        self.assertNotEqual(lang.stem_es("caballo"), lang.stem_es("caballero"))
        self.assertEqual(lang.stem_es("Corrió"), lang.stem_es("corrió"))
        # a passage is indexed under both spellings, so a question without accents still matches
        self.assertIn(lang.stem_es("corrio"), lang.stems("Marlow corrió hacia el faro", "es"))
        self.assertIn(lang.stem_es("corrió"), lang.stems("Marlow corrió hacia el faro", "es"))
        self.assertEqual(lang.stem_es("a"), "a")

    def test_language_detection(self):
        self.assertEqual(lang.detect(["El caballero salió de la venta y se fue por el camino con su escudero."]), "es")
        self.assertEqual(lang.detect(["The captain walked to the harbour and said that it was late."]), "en")
        self.assertEqual(lang.detect(["El " + PAD for _ in range(3)]), "es")
        self.assertEqual(lang.detect([""]), "en")

    def test_questions_drop_function_words_and_stem(self):
        self.assertEqual(lang.query_terms(lang.words("¿Quién es Lázaro y qué le dijo el ciego?"), "es"), ["lazar", "cieg"])
        # a question made only of function words still yields terms instead of an empty query
        self.assertEqual(len(lang.query_terms(["qué", "es"], "es")), 2)


class SpanishBookTests(unittest.TestCase):
    def setUp(self):
        self.library = Library(":memory:")
        self.library.add_work("Faro", LIBRO)
        self.addCleanup(self.library.close)
        self.h = [self.library.horizon_after_chapter("Faro", n) for n in (1, 2, 3)]

    def test_language_is_detected_and_stored(self):
        self.assertEqual(self.library.lang_of("Faro"), "es")

    def test_inflected_and_unaccented_questions_find_passages(self):
        found = self.library.search("Faro", "marineros que corren por el muelle", self.h[2])
        self.assertTrue(any("corrían" in p.text or "corrian" in p.text.lower() for p in found))
        self.assertTrue(any(p.chapter == 2 for p in self.library.search("Faro", "porque corrio el marinero", self.h[2])))
        self.assertTrue(self.library.search("Faro", "que farol colgaba en la bodega", self.h[1]))

    def test_nothing_after_the_reader_position_is_returned(self):
        for horizon, chapter in ((self.h[0], 1), (self.h[1], 2)):
            for question in ("¿Dónde está la llave de ámbar?", "¿Quién traicionó al capitán?", "tercer escalón traicionado"):
                for passage in self.library.search("Faro", question, horizon, limit=20):
                    self.assertLessEqual(passage.chapter, chapter)
                    self.assertNotIn("ámbar", passage.text)
        self.assertTrue(any("ámbar" in p.text for p in self.library.search("Faro", "llave de ámbar", self.h[2])))

    def test_locate_with_accents_and_dialogue_dashes(self):
        horizon, found = self.library.locate("Faro", "Nadie —respondió Inés mirando el mar")
        self.assertEqual(found.chapter, 1)
        horizon, found = self.library.locate("Faro", "los marineros corrían por el muelle porque la tormenta")
        self.assertEqual(found.chapter, 2)

    def test_reingest_replaces_the_spanish_index(self):
        self.library.add_work("Faro", LIBRO[:1])
        self.assertEqual(self.library.search("Faro", "llave de ámbar", 10**6), [])
        self.assertEqual(self.library.lang_of("Faro"), "es")

    def test_an_english_book_in_the_same_library_is_unaffected(self):
        english = [Chapter("One", ["The captain walked to the harbour and said that it was late. " * 6])]
        self.library.add_work("Harbour", english)
        self.assertEqual(self.library.lang_of("Harbour"), "en")
        self.assertTrue(self.library.search("Harbour", "walking captain", 10**6))
        self.assertTrue(self.library.search("Faro", "capitán", 10**6))

    def test_explicit_language_overrides_detection(self):
        self.library.add_work("Forzado", LIBRO, language="en")
        self.assertEqual(self.library.lang_of("Forzado"), "en")


class NamesAndMessagesTests(unittest.TestCase):
    def test_accented_names_and_titles_are_recognised(self):
        texts = [(i, f"Ayer vimos a doña Inés Álvarez junto al muelle y Inés Álvarez no dijo nada {i}.") for i in range(1, 4)]
        names = extract_names(texts)
        self.assertIn(("inés", "álvarez"), names)

    def test_name_alias_expansion_in_a_spanish_book(self):
        sancho = [Chapter("Capítulo I", [f"Ayer vimos a Sancho Panza cerca del molino, y Sancho Panza reía {i}." + PAD
                                          for i in range(4)]),
                  Chapter("Capítulo II", ["Luego Panza habló solo con el cura sobre el viaje largo." + PAD * 2])]
        library = Library(":memory:")
        self.addCleanup(library.close)
        library.add_work("Sancho", sancho)
        found = library.search("Sancho", "¿Qué hizo Sancho?", 10**6, limit=20)
        self.assertTrue(any("Panza habló" in p.text for p in found))

    def test_refusals_follow_the_language_of_the_question(self):
        self.assertTrue(is_future_question("¿Cómo acaba el libro?"))
        self.assertEqual(refusal("¿Cómo acaba el libro?"), REFUSAL_ES)
        self.assertNotEqual(refusal("How does it end?"), REFUSAL_ES)
        self.assertIn("Nada relevante", nothing_found("¿Quién es Lázaro?"))
        library = Library(":memory:")
        self.addCleanup(library.close)
        library.add_work("Faro", LIBRO)
        text, passages = answer.ask(library, "Faro", "¿Cómo acaba el libro?", 1, lambda q, p: self.fail("no model call"))
        self.assertEqual((text, passages), (REFUSAL_ES, []))


class SpanishHeadingTests(unittest.TestCase):
    def test_spanish_headings_letters_and_paratext(self):
        filler = " ".join(["La mañana era fría y el viento soplaba con fuerza sobre los campos de trigo y cebada."] * 4)
        raw = (f"Portada del libro\n\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\n"
               f"Prólogo\n\nEste prólogo del editor cuenta el final de la novela. {filler}\n\n"
               f"Capítulo primero. Que trata de la llegada al pueblo\n\n{filler}\n\n"
               f"Carta de don Fulano a su prima\n\n{filler}\n\n"                       # a letter inside chapter 1
               f"CAPÍTULO II\n\n{filler}\n\n"
               f"Capítulo III. De cómo se fue el caballero por el camino real hacia la ciudad de Toledo\n\n{filler}\n\n"
               f"*** END OF THE PROJECT GUTENBERG EBOOK X ***\n")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "libro.txt"
            path.write_text(raw, encoding="utf-8")
            chapters = read_book(path)
            self.assertEqual(len(chapters), 3)
            self.assertTrue(chapters[0].title.startswith("Capítulo primero"))
            self.assertTrue(any("Carta de don Fulano" in p for p in chapters[0].paragraphs))
            self.assertNotIn("final de la novela", " ".join(" ".join(c.paragraphs) for c in chapters))
            path.write_bytes(raw.encode("cp1252"))  # old Latin-1 style files still decode
            self.assertEqual(len(read_book(path)), 3)

    def test_cli_ingests_and_labels_the_language(self):
        with tempfile.TemporaryDirectory() as folder:
            book = Path(folder) / "faro.txt"
            book.write_text("\n\n".join(["Capítulo primero"] + [" ".join(p.paragraphs) for p in LIBRO[:1]]) + "\n\n"
                            + "Capítulo II\n\n" + " ".join(LIBRO[1].paragraphs), encoding="utf-8")
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(main(["--db", str(Path(folder) / "lib.db"), "--no-embeddings", "ingest", str(book), "--title", "Faro"]), 0)
            self.assertIn("(es)", out.getvalue())


if __name__ == "__main__":
    unittest.main()
