import re
import sys
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    List,
    Match,
    MutableMapping,
    Optional,
    Pattern,
    Type,
    TypeVar,
    Union,
    cast,
)

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

_LINE_END = re.compile(r"\n|$")


class BlockState:
    """The state to save block parser's cursor and tokens."""

    src: str
    tokens: List[Dict[str, Any]]
    cursor: int
    cursor_max: int
    list_tight: bool
    parent: Any
    env: MutableMapping[str, Any]

    def __init__(self, parent: Optional[Any] = None) -> None:
        self.src = ""
        self.tokens = []

        # current cursor position
        self.cursor = 0
        self.cursor_max = 0

        # for list and block quote chain
        self.list_tight = True
        self.parent = parent

        # for saving def references
        if parent:
            self.env = parent.env
        else:
            self.env = {"ref_links": {}}


class BaseRenderer(object):
    NAME: ClassVar[str] = "base"

    def __init__(self) -> None:
        self.__methods: Dict[str, Callable[..., str]] = {}

    def register(self, name: str, method: Callable[..., str]) -> None:
        """Register a render method for the named token. For example::

        def render_wiki(renderer, key, title):
            return f'<a href="/wiki/{key}">{title}</a>'

        renderer.register('wiki', render_wiki)
        """
        # bind self into renderer method
        self.__methods[name] = lambda *arg, **kwargs: method(self, *arg, **kwargs)

    def _get_method(self, name: str) -> Callable[..., str]:
        try:
            return cast(Callable[..., str], object.__getattribute__(self, name))
        except AttributeError:
            method = self.__methods.get(name)
            if not method:
                raise AttributeError('No renderer "{!r}"'.format(name))
            return method

    def render_token(self, token: Dict[str, Any], state: BlockState) -> str:
        func = self._get_method(token["type"])
        return func(token, state)

    def iter_tokens(self, tokens: Iterable[Dict[str, Any]], state: BlockState) -> Iterable[str]:
        for tok in tokens:
            yield self.render_token(tok, state)

    def render_tokens(self, tokens: Iterable[Dict[str, Any]], state: BlockState) -> str:
        return "".join(self.iter_tokens(tokens, state))

    def __call__(self, tokens: Iterable[Dict[str, Any]], state: BlockState) -> str:
        return self.render_tokens(tokens, state)
