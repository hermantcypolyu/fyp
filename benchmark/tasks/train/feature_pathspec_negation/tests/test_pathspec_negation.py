"""Hidden verification suite for feature_pathspec_negation.

Never mounted into the sandbox. Fourteen independent assertions so that a partial
implementation earns partial credit (AGENT_BUILD_SPEC 5.4 rule 5).

The first six pin the matching behaviour that already exists and must not regress; the
remaining eight cover negation and the explanation. ``explain`` is only ever reached as
an attribute of an instance, so its absence shows up as failing tests rather than as a
collection error that scores zero for everything.
"""

from src.pathspec import PathSpec

# --- existing behaviour: these pass before and after the feature lands ---------------


def test_a_bare_pattern_matches_a_basename_at_any_depth():
    spec = PathSpec(["*.log"])
    assert spec.is_ignored("debug.log") is True
    assert spec.is_ignored("var/log/debug.log") is True
    assert spec.is_ignored("debug.txt") is False


def test_a_question_mark_matches_exactly_one_character():
    spec = PathSpec(["draft?.md"])
    assert spec.is_ignored("draft1.md") is True
    assert spec.is_ignored("notes/draft7.md") is True
    assert spec.is_ignored("draft.md") is False
    assert spec.is_ignored("draft12.md") is False


def test_a_double_star_spans_any_number_of_directories():
    spec = PathSpec(["docs/**/notes.md"])
    assert spec.is_ignored("docs/notes.md") is True
    assert spec.is_ignored("docs/a/b/notes.md") is True
    assert spec.is_ignored("other/docs/notes.md") is False


def test_a_pattern_containing_a_slash_is_anchored_to_the_root():
    spec = PathSpec(["/build", "dist/out"])
    assert spec.is_ignored("build") is True
    assert spec.is_ignored("packages/build") is False
    assert spec.is_ignored("dist/out") is True
    assert spec.is_ignored("a/dist/out") is False


def test_a_trailing_slash_matches_directories_and_their_contents_only():
    spec = PathSpec(["node_modules/"])
    assert spec.is_ignored("node_modules/") is True
    assert spec.is_ignored("node_modules/left-pad/index.js") is True
    assert spec.is_ignored("src/node_modules/bundle.js") is True
    assert spec.is_ignored("node_modules") is False


def test_blank_lines_and_comments_are_not_patterns():
    spec = PathSpec(["# build output", "", "   ", "*.tmp", "  *.bak  "])
    assert len(spec.rules) == 2
    assert spec.is_ignored("scratch.tmp") is True
    assert spec.is_ignored("notes/old.bak") is True
    assert spec.is_ignored("build output") is False


# --- the feature ---------------------------------------------------------------------


def test_a_later_bang_pattern_re_includes_a_file():
    spec = PathSpec(["*.log", "!keep.log"])
    assert spec.is_ignored("debug.log") is True
    assert spec.is_ignored("keep.log") is False
    assert spec.is_ignored("var/keep.log") is False


def test_the_last_matching_pattern_wins():
    spec = PathSpec(["!keep.log", "*.log"])
    assert spec.is_ignored("keep.log") is True
    assert spec.explain("keep.log") == "*.log"
    reordered = PathSpec(["*.log", "!keep.log"])
    assert reordered.is_ignored("keep.log") is False
    assert reordered.explain("keep.log") == "!keep.log"


def test_a_bang_pattern_decides_even_with_nothing_to_undo():
    spec = PathSpec(["!keep.log"])
    assert spec.is_ignored("keep.log") is False
    assert spec.explain("keep.log") == "!keep.log"


def test_a_bang_cannot_rescue_a_file_under_an_excluded_directory():
    spec = PathSpec(["build/", "!build/keep.txt"])
    assert spec.is_ignored("build/keep.txt") is True
    assert spec.explain("build/keep.txt") == "build/"


def test_re_including_a_directory_lets_the_walk_reach_its_contents():
    spec = PathSpec(["logs/", "!logs/", "logs/*.tmp"])
    assert spec.is_ignored("logs/a.txt") is False
    assert spec.is_ignored("logs/b.tmp") is True
    assert spec.explain("logs/b.tmp") == "logs/*.tmp"


def test_explain_names_the_deciding_pattern_verbatim():
    spec = PathSpec(["*.log", "!var/keep.log", "var/tmp/"])
    assert spec.explain("debug.log") == "*.log"
    assert spec.explain("var/keep.log") == "!var/keep.log"
    assert spec.explain("var/tmp/scratch") == "var/tmp/"


def test_explain_returns_none_when_no_pattern_matched():
    spec = PathSpec(["*.log", "!keep.log"])
    assert spec.explain("src/main.py") is None
    assert spec.is_ignored("src/main.py") is False


def test_a_backslash_escapes_a_leading_bang():
    spec = PathSpec(["*.log", "\\!important.log"])
    assert spec.is_ignored("!important.log") is True
    assert spec.explain("!important.log") == "\\!important.log"
    assert spec.is_ignored("important.log") is True
