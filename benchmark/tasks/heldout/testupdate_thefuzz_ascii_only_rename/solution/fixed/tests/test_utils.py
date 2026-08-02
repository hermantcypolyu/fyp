import re
import unittest

from thefuzz import utils
from thefuzz.string_processing import StringProcessor


class StringProcessingTest(unittest.TestCase):
    def test_replace_non_letters_non_numbers_with_whitespace(self):
        strings = ["new york mets - atlanta braves", "Cães danados",
                   "New York //// Mets $$$", "Ça va?"]
        for string in strings:
            proc_string = StringProcessor.replace_non_letters_non_numbers_with_whitespace(string)
            regex = re.compile(r"(?ui)[\W]")
            for expr in regex.finditer(proc_string):
                self.assertEqual(expr.group(), " ")

    def test_dont_condense_whitespace(self):
        s1 = "new york mets - atlanta braves"
        s2 = "new york mets atlanta braves"
        p1 = StringProcessor.replace_non_letters_non_numbers_with_whitespace(s1)
        p2 = StringProcessor.replace_non_letters_non_numbers_with_whitespace(s2)
        self.assertNotEqual(p1, p2)


class UtilsTest(unittest.TestCase):
    def setUp(self):
        self.s1 = "new york mets"
        self.s1a = "new york mets"
        self.s2 = "new YORK mets"
        self.s3 = "the wonderful new york mets"
        self.s4 = "new york mets vs atlanta braves"
        self.s5 = "atlanta braves vs new york mets"
        self.s6 = "new york mets - atlanta braves"
        self.mixed_strings = [
            "Lorem Ipsum is simply dummy text of the printing and typesetting industry.",
            "C'est la vie",
            "Ça va?",
            "Cães danados",
            "\xacCamarões assados",
            "a\xac\u1234\u20ac\U00008000",
            "\u00C1"
        ]

    def tearDown(self):
        pass

    def test_ascii_only(self):
        for s in self.mixed_strings:
            utils.ascii_only(s)

    def test_fullProcess(self):
        for s in self.mixed_strings:
            utils.full_process(s)

    def test_fullProcessForceAscii(self):
        for s in self.mixed_strings:
            utils.full_process(s, force_ascii=True)

    def test_make_type_consistent(self):
        s1, s2 = utils.make_type_consistent(self.s1, str(self.s2))
        self.assertIsInstance(s1, str)
        self.assertIsInstance(s2, str)


if __name__ == '__main__':
    unittest.main()
