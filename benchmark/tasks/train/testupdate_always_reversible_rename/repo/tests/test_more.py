from itertools import count
from unittest import TestCase

import more_itertools as mi


class OneTests(TestCase):
    def test_basic(self):
        it = iter(['item'])
        self.assertEqual(mi.one(it), 'item')

    def test_too_short(self):
        it = iter([])
        self.assertRaises(ValueError, lambda: mi.one(it))
        self.assertRaises(IndexError, lambda: mi.one(it, too_short=IndexError))

    def test_too_long(self):
        it = count()
        self.assertRaises(ValueError, lambda: mi.one(it))  # burn 0 and 1
        self.assertEqual(next(it), 2)
        self.assertRaises(
            OverflowError, lambda: mi.one(it, too_long=OverflowError)
        )


class UniqueToEachTests(TestCase):
    """Tests for ``unique_to_each()``"""

    def test_all_unique(self):
        """When all the input iterables are unique the output should match
        the input."""
        iterables = [[1, 2], [3, 4, 5], [6, 7, 8]]
        self.assertEqual(mi.unique_to_each(*iterables), iterables)

    def test_duplicates(self):
        """When there are duplicates in any of the input iterables that aren't
        in the rest, those duplicates should be emitted."""
        iterables = ["mississippi", "missouri"]
        self.assertEqual(
            mi.unique_to_each(*iterables), [['p', 'p'], ['o', 'u', 'r']]
        )

    def test_mixed(self):
        """When the input iterables contain different types the function should
        still behave properly"""
        iterables = ['x', (i for i in range(3)), [1, 2, 3], tuple()]
        self.assertEqual(mi.unique_to_each(*iterables), [['x'], [0], [3], []])


class SortTogetherTest(TestCase):
    """Tests for sort_together()"""

    def test_key_list(self):
        """tests `key_list` including default, iterables include duplicates"""
        iterables = [
            ['GA', 'GA', 'GA', 'CT', 'CT', 'CT'],
            ['May', 'Aug.', 'May', 'June', 'July', 'July'],
            [97, 20, 100, 70, 100, 20]
        ]

        self.assertEqual(
            mi.sort_together(iterables),
            [
                ('CT', 'CT', 'CT', 'GA', 'GA', 'GA'),
                ('June', 'July', 'July', 'May', 'Aug.', 'May'),
                (70, 100, 20, 97, 20, 100)
            ]
        )

        self.assertEqual(
            mi.sort_together(iterables, key_list=(0, 1)),
            [
                ('CT', 'CT', 'CT', 'GA', 'GA', 'GA'),
                ('July', 'July', 'June', 'Aug.', 'May', 'May'),
                (100, 20, 70, 20, 97, 100)
            ]
        )

    def test_reverse(self):
        iterables = [[4, 3, 2, 1], ['a', 'b', 'c', 'd']]
        self.assertEqual(
            mi.sort_together(iterables, reverse=True),
            [(4, 3, 2, 1), ('a', 'b', 'c', 'd')]
        )


class DivideTest(TestCase):
    """Tests for divide()"""

    def test_invalid_n(self):
        self.assertRaises(ValueError, lambda: mi.divide(-1, [1, 2, 3]))
        self.assertRaises(ValueError, lambda: mi.divide(0, [1, 2, 3]))

    def test_basic(self):
        iterable = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        for n, expected in [
            (1, [iterable]),
            (2, [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
            (3, [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10]]),
            (10, [[n] for n in range(1, 10 + 1)]),
        ]:
            self.assertEqual(
                [list(x) for x in mi.divide(n, iterable)], expected
            )

    def test_large_n(self):
        iterable = [1, 2, 3, 4]
        self.assertEqual(
            [list(x) for x in mi.divide(6, iterable)],
            [[1], [2], [3], [4], [], []]
        )


class TestAlwaysIterable(TestCase):
    """Tests for always_iterable()"""
    def test_single(self):
        self.assertEqual(list(mi.always_iterable(1)), [1])

    def test_strings(self):
        for obj in ['foo', b'bar', u'baz']:
            actual = list(mi.always_iterable(obj))
            expected = [obj]
            self.assertEqual(actual, expected)

    def test_base_type(self):
        dict_obj = {'a': 1, 'b': 2}
        str_obj = '123'

        # Default: dicts are iterable like they normally are
        default_actual = list(mi.always_iterable(dict_obj))
        default_expected = list(dict_obj)
        self.assertEqual(default_actual, default_expected)

        # Unitary types set: dicts are not iterable
        custom_actual = list(mi.always_iterable(dict_obj, base_type=dict))
        custom_expected = [dict_obj]
        self.assertEqual(custom_actual, custom_expected)

        # With unitary types set, strings are iterable
        str_actual = list(mi.always_iterable(str_obj, base_type=None))
        str_expected = list(str_obj)
        self.assertEqual(str_actual, str_expected)


class ReversedExtendedTests(TestCase):
    """Tests for ``reversed_extended()``"""

    def test_regular_reversed(self):
        self.assertEqual(list(reversed(range(10))),
                         list(mi.reversed_extended(range(10))))
        self.assertEqual(list(reversed([1, 2, 3])),
                         list(mi.reversed_extended([1, 2, 3])))
        self.assertEqual(reversed([1, 2, 3]).__class__,
                         mi.reversed_extended([1, 2, 3]).__class__)

    def test_nonseq_reversed(self):
        # Create a non-reversible generator from a sequence
        with self.assertRaises(TypeError):
            reversed(x for x in range(10))

        self.assertEqual(list(reversed(range(10))),
                         list(mi.reversed_extended(x for x in range(10))))
        self.assertEqual(list(reversed([1, 2, 3])),
                         list(mi.reversed_extended(x for x in [1, 2, 3])))
        self.assertNotEqual(reversed([1, 2]).__class__,
                            mi.reversed_extended(x for x in [1, 2]).__class__)
