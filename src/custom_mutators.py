# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Named custom mutation strategies and the shared fuzzing dictionary.

Atheris has always accepted a single ``custom_mutator`` callable through
``atheris.Setup()``.  This module adds two related pieces of infrastructure:

* a process-wide registry of named mutation strategies that can be selected by
  name when the fuzzer starts, and
* a dictionary that is parsed exactly once, at startup, and shared by the
  selected strategy and the built-in mutation path (``atheris.Mutate``).

The registry and the selected dictionary are frozen the first time a
``Setup()`` call succeeds.  Modifying either one afterwards fails without
changing the configuration that is already in effect.

Dictionary format
-----------------
One entry per line.  Blank lines and lines whose first non-space character is
``#`` are ignored.  The canonical libFuzzer form is a double-quoted entry::

    "GET"
    "\\x00\\x01"

For convenience an unquoted line is also accepted and means the same thing::

    GET

In both forms ``\\\\``, ``\\"``, ``\\xAB`` and the usual C escapes (``\\n``,
``\\r``, ``\\t``, ``\\0``) are decoded.  Entries keep their file order; that
order is what makes dictionary-assisted mutation sequences reproducible for a
fixed ``-seed`` and is exposed by :meth:`Dictionary.format_entries`.
"""

from __future__ import annotations

import atexit
import hashlib
import os
import sys
import tempfile
import traceback
from collections.abc import Callable, Iterable
from typing import Any

# libFuzzer stores each dictionary entry in a 64-byte Word and keeps at most
# 2**14 entries.  Applying the same limits here means a dictionary accepted by
# Atheris is also accepted by the engine.
MAX_DICTIONARY_ENTRY_SIZE = 64
MAX_DICTIONARY_SIZE = 1 << 14

# A strategy is anything callable with (data, max_size, seed) -> bytes.
CustomMutator = Callable[[bytes, int, int], bytes]

_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_SIMPLE_ESCAPES = {
    "\\": 0x5C,
    '"': 0x22,
    "n": 0x0A,
    "r": 0x0D,
    "t": 0x09,
    "0": 0x00,
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AtherisSetupError(RuntimeError):
  """Raised when the fuzzer cannot be configured; startup is aborted."""


class UnknownCustomMutatorError(AtherisSetupError):
  """Raised when the strategy selected by name has not been registered."""


class RegistryFrozenError(RuntimeError):
  """Raised when the strategy registry is modified after startup."""


class CustomMutatorError(RuntimeError):
  """Base class for errors raised while invoking a custom mutation strategy."""


class CustomMutatorContractError(CustomMutatorError):
  """A strategy violated the ``(data, max_size, seed) -> bytes`` contract."""


class CustomMutatorOutputTooLongError(CustomMutatorContractError):
  """A strategy returned more bytes than ``max_size`` allows."""


class CustomMutatorTypeError(CustomMutatorContractError, TypeError):
  """A strategy returned something other than ``bytes``."""


class CustomMutatorRuntimeError(CustomMutatorError):
  """A strategy raised; carries the strategy name and original traceback."""


class DictionaryError(AtherisSetupError):
  """The dictionary file is missing, unreadable, or contains a bad entry."""

  def __init__(self, path: str, message: str,
               line_number: int | None = None):
    self.path = path
    self.line_number = line_number
    self.reason = message
    if line_number is None:
      super().__init__(f"Dictionary {path!r}: {message}")
    else:
      super().__init__(f"Dictionary {path!r} line {line_number}: {message}")


# ---------------------------------------------------------------------------
# Dictionary parsing
# ---------------------------------------------------------------------------


def _decode_body(body: str, path: str, line_number: int,
                 allow_literal_quote: bool) -> bytes:
  """Decodes the body of an entry (the text without its surrounding quotes)."""
  decoded = bytearray()
  index = 0
  length = len(body)
  while index < length:
    char = body[index]
    if char == "\\":
      if index + 1 >= length:
        raise DictionaryError(path, "invalid escape at end of entry",
                              line_number)
      escaped = body[index + 1]
      if escaped in _SIMPLE_ESCAPES:
        decoded.append(_SIMPLE_ESCAPES[escaped])
        index += 2
        continue
      if escaped == "x":
        hex_digits = body[index + 2:index + 4]
        if len(hex_digits) == 2 and all(c in _HEX_DIGITS for c in hex_digits):
          decoded.append(int(hex_digits, 16))
          index += 4
          continue
        raise DictionaryError(
            path, "invalid \\x escape (expected two hexadecimal digits)",
            line_number)
      raise DictionaryError(
          path,
          f"invalid escape '\\{escaped}' "
          "(only \\\\, \\\", \\n, \\r, \\t, \\0 and \\xAB are supported)",
          line_number)
    if char == '"' and not allow_literal_quote:
      raise DictionaryError(
          path, "unescaped '\"' in entry (use \\\" for a literal quote)",
          line_number)
    codepoint = ord(char)
    if codepoint > 0xFF:
      raise DictionaryError(
          path,
          f"non-ASCII character {char!r} in entry "
          "(use \\xAB escapes for non-ASCII bytes)", line_number)
    decoded.append(codepoint)
    index += 1

  if len(decoded) > MAX_DICTIONARY_ENTRY_SIZE:
    raise DictionaryError(
        path,
        f"entry is {len(decoded)} bytes, the maximum is "
        f"{MAX_DICTIONARY_ENTRY_SIZE}", line_number)
  return bytes(decoded)


def parse_dictionary_entry(line: str,
                           path: str = "<dictionary>",
                           line_number: int = 0) -> bytes | None:
  """Parses one line of a dictionary file.

  Returns ``None`` for blank and comment lines and raises
  :class:`DictionaryError` (naming the line) for anything malformed.

  Two forms are accepted:

  * the libFuzzer form, ``"value"`` or ``key="value"`` -- the text from the
    first quote to the last quote is the entry;
  * a bare ``value`` line, where the trimmed line is the entry.
  """
  stripped = line.strip()
  if not stripped:
    return None
  if stripped.startswith("#"):
    return None

  if '"' in stripped:
    start = stripped.find('"')
    end = len(stripped) - 1
    if stripped[end] != '"' or end == start:
      raise DictionaryError(path, "unterminated dictionary entry", line_number)
    if end == start + 1:
      raise DictionaryError(path, "dictionary entry must not be empty",
                            line_number)
    return _decode_body(stripped[start + 1:end], path, line_number,
                        allow_literal_quote=False)

  # Bare form: no quoting was used, so a quote would be a literal byte, but by
  # construction there is none here.
  return _decode_body(stripped, path, line_number, allow_literal_quote=True)


def parse_dictionary(text: str,
                     path: str = "<dictionary>") -> list[bytes]:
  """Parses dictionary file contents into an ordered list of entries.

  Blank lines and comments are skipped.  The first malformed line aborts
  parsing with a :class:`DictionaryError` that includes its 1-based line
  number.  Entries are returned in file order.
  """
  entries: list[bytes] = []
  # Split on "\n" only: that is what libFuzzer does, and it keeps line numbers
  # accurate for files with "\r\n" endings.
  for line_number, raw_line in enumerate(text.split("\n"), start=1):
    line = raw_line.removesuffix("\r")
    entry = parse_dictionary_entry(line, path, line_number)
    if entry is None:
      continue
    entries.append(entry)
    if len(entries) > MAX_DICTIONARY_SIZE:
      raise DictionaryError(
          path,
          f"more than {MAX_DICTIONARY_SIZE} entries (the engine limit)",
          line_number)
  return entries


class Dictionary:
  """An immutable, ordered set of dictionary entries.

  A single instance is shared by every consumer -- the built-in mutation path,
  ``atheris.Mutate()`` and custom strategies -- so strategies never keep their
  own copies of the words.  The entry sequence is a tuple, and both the tuple
  and the path are exposed as read-only properties so callers cannot replace
  what the engine has already been started with.
  """

  __slots__ = ("_digest", "_entries", "_path")

  def __init__(self,
               entries: Iterable[bytes],
               path: str = "<memory>"):
    normalized = []
    for entry in entries:
      if not isinstance(entry, (bytes, bytearray)):
        raise TypeError(
            "Dictionary entries must be bytes, "
            f"got {type(entry).__name__}")
      entry = bytes(entry)
      if len(entry) > MAX_DICTIONARY_ENTRY_SIZE:
        raise DictionaryError(
            path,
            f"entry is {len(entry)} bytes, the maximum is "
            f"{MAX_DICTIONARY_ENTRY_SIZE}")
      normalized.append(entry)
    if len(normalized) > MAX_DICTIONARY_SIZE:
      raise DictionaryError(
          path, f"more than {MAX_DICTIONARY_SIZE} entries (the engine limit)")
    object.__setattr__(self, "_entries", tuple(normalized))
    object.__setattr__(self, "_path", path)
    object.__setattr__(
        self, "_digest", hashlib.sha256(b"\x00".join(normalized)).hexdigest())

  def __setattr__(self, name, value):
    raise AttributeError(f"Dictionary is immutable; cannot set {name!r}")

  def __delattr__(self, name):
    raise AttributeError(f"Dictionary is immutable; cannot delete {name!r}")

  @property
  def entries(self) -> tuple[bytes, ...]:
    return self._entries

  @property
  def path(self) -> str:
    return self._path

  @classmethod
  def from_file(cls, path: str | os.PathLike) -> Dictionary:
    path = os.fspath(path)
    try:
      with open(path, "rb") as dictionary_file:
        raw = dictionary_file.read()
    except FileNotFoundError as exc:
      raise DictionaryError(path, "file does not exist") from exc
    except IsADirectoryError as exc:
      raise DictionaryError(path, "is a directory, not a file") from exc
    except OSError as exc:
      raise DictionaryError(
          path, f"cannot be read: {exc.strerror}") from exc
    try:
      text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
      raise DictionaryError(
          path,
          f"is not valid UTF-8 (byte offset {exc.start})") from exc
    return cls(parse_dictionary(text, path), path)

  def __len__(self) -> int:
    return len(self.entries)

  def __iter__(self):
    return iter(self.entries)

  def __getitem__(self, index):
    return self.entries[index]

  def __eq__(self, other) -> bool:
    if isinstance(other, Dictionary):
      return self._digest == other._digest
    return NotImplemented

  def __hash__(self) -> int:
    return hash(self._digest)

  def __repr__(self) -> str:
    return f"Dictionary({self.entries!r}, path={self.path!r})"

  @property
  def digest(self) -> str:
    return self._digest

  def describe(self) -> str:
    """A one-line summary suitable for startup logs."""
    count = len(self.entries)
    return (f"Dictionary: {count} entr"
            f"{'y' if count == 1 else 'ies'} from {self.path}")

  def format_entries(self) -> str:
    """A stable, printable rendering of every entry, in file order.

    Order is part of the reproducibility contract, so this output can be
    compared between runs and seeds.
    """
    lines = [self.describe()]
    for index, entry in enumerate(self.entries):
      lines.append(f"  [{index:04d}] {entry!r}")
    return "\n".join(lines)

  def write_libfuzzer_file(self, path: str | os.PathLike) -> None:
    """Writes the entries in libFuzzer's on-disk format.

    The engine is handed this serialization of the already-parsed object, so
    both mutation paths see the same bytes in the same order.
    """
    with open(os.fspath(path), "w", encoding="ascii", newline="\n") as output:
      for entry in self.entries:
        escaped = bytearray()
        for byte in entry:
          if byte in (0x5C, 0x22):  # backslash, double quote
            escaped.append(0x5C)
            escaped.append(byte)
          elif 0x20 <= byte <= 0x7E:
            escaped.append(byte)
          else:
            escaped.extend(f"\\x{byte:02x}".encode("ascii"))
        output.write('"' + escaped.decode("ascii") + '"\n')


def load_dictionary(path: str | os.PathLike) -> Dictionary:
  """Parses a dictionary file eagerly.

  Raises :class:`DictionaryError` (an :class:`AtherisSetupError`) with a line
  number if the file is missing, unreadable, or malformed.
  """
  return Dictionary.from_file(path)


def parse_dictionary_file(path: str | os.PathLike) -> Dictionary:
  """Alias for :func:`load_dictionary`."""
  return load_dictionary(path)


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------


class MutationStrategy:
  """A registered custom mutation strategy.

  Attributes:
    name: the name used to select the strategy at startup.
    mutator: the callable implementing the strategy.
    requires_dictionary: when true, ``Setup()`` refuses to start unless a
      dictionary was supplied.
  """

  __slots__ = ("__weakref__", "mutator", "name", "requires_dictionary")

  def __init__(self, name: str, mutator: CustomMutator,
               requires_dictionary: bool = False):
    self.name = name
    self.mutator = mutator
    self.requires_dictionary = requires_dictionary

  def __repr__(self) -> str:
    return (f"MutationStrategy(name={self.name!r}, "
            f"requires_dictionary={self.requires_dictionary!r})")

  def __call__(self, data: bytes, max_size: int, seed: int) -> bytes:
    return self.mutator(data, max_size, seed)


_registry: dict[str, MutationStrategy] = {}
_frozen = False
_frozen_reason = ""


def is_frozen() -> bool:
  """True once ``Setup()`` has committed the registry and dictionary."""
  return _frozen


def freeze(reason: str = "Atheris.Setup() has already been called") -> None:
  """Freezes registration.  Called by ``Setup()``; not public API."""
  global _frozen, _frozen_reason
  _frozen = True
  _frozen_reason = reason


def _reset_for_testing() -> None:
  """Restores a pristine process state.  Tests only; never used in fuzzing."""
  global _frozen, _frozen_reason, _active_invoker
  _frozen = False
  _frozen_reason = ""
  _registry.clear()
  _process_dictionary.set(None)
  _active_invoker = None


def register_custom_mutator(
    name: str,
    mutator: CustomMutator | None = None,
    *,
    requires_dictionary: bool = False,
) -> Any:
  """Registers a named custom mutation strategy.

  Registration is only possible before the fuzzer starts.  Once ``Setup()``
  has succeeded the registry is frozen and further calls raise
  :class:`RegistryFrozenError` without modifying anything.

  Args:
    name: a non-empty name used to select this strategy.
    mutator: a callable with the signature ``(data, max_size, seed) -> bytes``.
      May be omitted when the function is used as a decorator.
    requires_dictionary: if true, ``Setup()`` refuses to start when no
      dictionary is supplied.

  Returns:
    The registered :class:`MutationStrategy`, or a decorator when ``mutator``
    is omitted.
  """
  if not isinstance(name, str) or not name:
    raise ValueError("Custom mutator name must be a non-empty string")
  if _frozen:
    raise RegistryFrozenError(
        f"Cannot register custom mutator {name!r}: {_frozen_reason}. "
        "The mutation strategy registry is fixed once startup begins.")
  if mutator is None:

    def decorator(func: CustomMutator) -> CustomMutator:
      register_custom_mutator(
          name, func, requires_dictionary=requires_dictionary)
      return func

    return decorator
  if name in _registry:
    raise ValueError(
        f"Custom mutator {name!r} is already registered; choose a different "
        "name")
  if not callable(mutator):
    raise TypeError(
        "Custom mutator must be callable with (data, max_size, seed)")

  strategy = MutationStrategy(name, mutator, requires_dictionary)
  _registry[name] = strategy
  return strategy


def register(name: str, *,
             requires_dictionary: bool = False) -> Callable[[CustomMutator],
                                                            CustomMutator]:
  """Decorator form of :func:`register_custom_mutator`."""
  return register_custom_mutator(
      name, None, requires_dictionary=requires_dictionary)


def register_custom_mutators(strategies: Any) -> list[MutationStrategy]:
  """Registers several strategies at once.

  ``strategies`` may be a mapping of name to callable or an iterable of
  ``(name, callable)`` pairs.  Registration is all-or-nothing: if any entry is
  invalid or the registry is frozen, nothing is registered.
  """
  if _frozen:
    raise RegistryFrozenError(
        f"Cannot register custom mutators: {_frozen_reason}. "
        "The mutation strategy registry is fixed once startup begins.")
  items = strategies.items() if hasattr(strategies, "items") else strategies
  items = list(items)

  # Validate everything before touching the registry.
  seen = set()
  validated = []
  for item in items:
    try:
      name, mutator = item
    except (TypeError, ValueError):
      raise TypeError(
          "register_custom_mutators expects (name, callable) pairs, "
          f"got {item!r}") from None
    if not isinstance(name, str) or not name:
      raise ValueError("Custom mutator name must be a non-empty string")
    if name in seen or name in _registry:
      raise ValueError(
          f"Custom mutator {name!r} is already registered; choose a different "
          "name")
    if not callable(mutator):
      raise TypeError(
          f"Custom mutator {name!r} must be callable with "
          "(data, max_size, seed)")
    seen.add(name)
    validated.append((name, mutator))

  return [register_custom_mutator(name, mutator) for name, mutator in validated]


def unregister_custom_mutator(name: str) -> None:
  """Removes a registered strategy.  Only allowed before startup."""
  if _frozen:
    raise RegistryFrozenError(
        f"Cannot unregister custom mutator {name!r}: {_frozen_reason}.")
  try:
    del _registry[name]
  except KeyError:
    raise KeyError(f"No custom mutator named {name!r} is registered") from None


def registered_names() -> list[str]:
  """The registered strategy names, in registration order."""
  return list(_registry)


def registered_custom_mutators() -> dict[str, MutationStrategy]:
  """A copy of the name-to-strategy mapping, in registration order."""
  return dict(_registry)


def get_registered(name: str) -> MutationStrategy:
  """Looks up a registered strategy or raises with the list of options."""
  try:
    return _registry[name]
  except KeyError:
    available = registered_names()
    if available:
      options = ", ".join(repr(option) for option in available)
    else:
      options = "<none>"
    raise UnknownCustomMutatorError(
        f"No custom mutator named {name!r} is registered. "
        f"Available custom mutators: {options}") from None


def get_custom_mutator(name: str) -> CustomMutator:
  """The callable registered under ``name``."""
  return get_registered(name).mutator


def _resolve_strategy(
    selector: None | str | MutationStrategy | CustomMutator
) -> MutationStrategy | None:
  """Turns a registered name, a MutationStrategy, or a callable into one."""
  if selector is None:
    return None
  if isinstance(selector, str):
    return get_registered(selector)
  if isinstance(selector, MutationStrategy):
    return selector
  if callable(selector):
    return MutationStrategy(
        getattr(selector, "__name__", "custom_mutator"),
        selector,
        bool(getattr(selector, "requires_dictionary", False)))
  raise TypeError(
      "custom_mutator must be a registered name, a MutationStrategy, or a "
      f"callable with (data, max_size, seed); got {type(selector).__name__}")


def resolve_strategy_for_setup(
    selector: None | str | MutationStrategy | CustomMutator
) -> MutationStrategy | None:
  """Public wrapper used by ``atheris.Setup()`` to resolve a selector."""
  return _resolve_strategy(selector)


# ---------------------------------------------------------------------------
# Invocation and contract enforcement
# ---------------------------------------------------------------------------


def _format_exception_with_traceback(exc: BaseException) -> str:
  return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


class StrategyInvoker:
  """Wraps a user strategy with the engine's contract checks.

  This is the object installed into the native mutation callback.  It enforces
  the byte-string and length contracts **before** the engine copies the result
  into its input buffer, and annotates failures with the strategy name and the
  original traceback.

  Two entry points are provided:

  * :meth:`invoke` always raises to its caller.  This is the direct,
    caller-visible API (see :func:`invoke_custom_mutator`) and leaves the
    process alive.
  * ``__call__`` is the native engine callback.  Recoverable contract
    violations (oversized or non-``bytes`` results) are reported once per kind
    and turned into an empty result, which the engine treats as a rejected
    mutation, so fuzzing continues.  A strategy that raises is not swallowed:
    it propagates to the native boundary, which reports it (with the strategy
    name and traceback already attached here) and stops the run.

  ``dictionary_aware_mutate`` records whether a generic mutation requested from
  inside the strategy sees the dictionary.  That decision is made once, at
  startup, and printed by :func:`DescribeMutationConfiguration`.
  """

  __slots__ = (
    "_reported",
    "active",
    "contract_violations",
    "dictionary",
    "recover_contract_errors",
    "strategy",
  )

  def __init__(self, strategy: MutationStrategy,
               dictionary: Dictionary | None,
               recover_contract_errors: bool = True):
    self.strategy = strategy
    self.dictionary = dictionary
    self.active = False
    self._reported = set()
    self.contract_violations = 0
    # When true, ``__call__`` (the engine callback) converts recoverable
    # contract violations into a rejected mutation so the process survives.
    # ``invoke`` always raises to its caller; that is the direct API.
    self.recover_contract_errors = recover_contract_errors

  @property
  def dictionary_aware_mutate(self) -> bool:
    return self.dictionary is not None

  def __call__(self, data: bytes, max_size: int, seed: int) -> bytes:
    """Engine callback entry point.

    With ``recover_contract_errors`` (the default) contract violations are
    reported once per error class and converted into ``b""`` so they never
    cross the C/C++ boundary as exceptions; the engine treats that as a
    rejected mutation and keeps running.  Either way the strategy's own
    exceptions are not swallowed.
    """
    try:
      return self.invoke(data, max_size, seed)
    except CustomMutatorContractError as exc:
      if not self.recover_contract_errors:
        raise
      self.contract_violations += 1
      if exc.__class__ not in self._reported:
        self._reported.add(exc.__class__)
        sys.stderr.write(f"WARNING: {exc}\n")
        sys.stderr.write(
            "WARNING: the mutation was rejected; fuzzing continues "
            "(subsequent violations of this kind are counted, not repeated).\n")
        sys.stderr.flush()
      return b""

  def invoke(self, data: bytes, max_size: int, seed: int) -> bytes:
    """Direct invocation; every failure raises to the caller."""
    if not isinstance(data, (bytes, bytearray)):
      raise CustomMutatorTypeError(
          "Atheris internal error: custom mutator was called with "
          f"{type(data).__name__}, expected bytes")
    data = bytes(data)

    previous = _enter_invoker(self)
    try:
      try:
        result = self.strategy.mutator(data, max_size, seed)
      except CustomMutatorError:
        raise
      except BaseException as exc:  # pylint: disable=broad-except
        raise CustomMutatorRuntimeError(
            f"Custom mutator {self.strategy.name!r} raised "
            f"{type(exc).__name__}: {exc}\n"
            "Original traceback:\n" +
            _format_exception_with_traceback(exc)) from exc
    finally:
      _leave_invoker(previous)

    return self._validate(result, max_size)

  def _validate(self, result: Any, max_size: int) -> bytes:
    if not isinstance(result, bytes):
      raise CustomMutatorTypeError(
          f"Custom mutator {self.strategy.name!r} must return bytes, "
          f"got {type(result).__name__} (expected bytes)")
    if len(result) > max_size:
      raise CustomMutatorOutputTooLongError(
          f"Custom mutator {self.strategy.name!r} returned {len(result)} "
          f"bytes, which exceeds max_size={max_size}; the data was rejected "
          "before it reached the engine buffer.")
    return result


_active_invoker: StrategyInvoker | None = None


def _enter_invoker(invoker: StrategyInvoker) -> StrategyInvoker | None:
  global _active_invoker
  previous = _active_invoker
  _active_invoker = invoker
  invoker.active = True
  return previous


def _leave_invoker(previous: StrategyInvoker | None) -> None:
  global _active_invoker
  if _active_invoker is not None:
    _active_invoker.active = False
  _active_invoker = previous


def set_active_invoker(invoker: StrategyInvoker | None) -> None:
  """Installs the invoker used for configuration reporting and introspection."""
  global _active_invoker
  _active_invoker = invoker


def get_active_invoker() -> StrategyInvoker | None:
  return _active_invoker


def dictionary_aware_mutate() -> bool:
  """True when a generic mutation issued *now* should consult the dictionary.

  The answer is fixed at startup: generic mutations performed by
  ``atheris.Mutate()`` from inside a strategy use the engine's manual
  dictionary, which Atheris populates from the same parsed entries that
  strategies see through :func:`get_dictionary`.
  """
  return (_active_invoker is not None and
          _active_invoker.dictionary_aware_mutate and
          _active_invoker.active)


def invoke_custom_mutator(strategy: str | MutationStrategy | CustomMutator,
                          data: bytes, max_size: int, seed: int) -> bytes:
  """Invokes a strategy synchronously, enforcing the mutation contract.

  This is the Python-visible call path.  A non-``bytes`` result or an oversized
  result raises a :class:`CustomMutatorContractError`; an exception raised by
  the strategy is wrapped in :class:`CustomMutatorRuntimeError` with the
  strategy name and the original traceback.  In every case the exception is
  catchable and the process keeps running.

  Args:
    strategy: a registered name, a :class:`MutationStrategy`, or a callable.
    data: the input to mutate.
    max_size: the maximum number of bytes the strategy may return.
    seed: the mutation seed.

  Returns:
    The mutated bytes.
  """
  if isinstance(strategy, str):
    strategy = get_registered(strategy)
  if isinstance(strategy, MutationStrategy):
    invoker = StrategyInvoker(strategy, _process_dictionary.get())
  elif callable(strategy):
    invoker = StrategyInvoker(
        MutationStrategy(getattr(strategy, "__name__", "custom_mutator"),
                         strategy), _process_dictionary.get())
  else:
    raise TypeError(
        "strategy must be a registered name, MutationStrategy, or callable, "
        f"got {type(strategy).__name__}")
  return invoker.invoke(data, max_size, seed)


def invoke_registered_mutator(name: str, data: bytes, max_size: int,
                              seed: int) -> bytes:
  """Alias for ``invoke_custom_mutator(name, ...)``."""
  return invoke_custom_mutator(name, data, max_size, seed)


# ---------------------------------------------------------------------------
# Process-wide dictionary
# ---------------------------------------------------------------------------

# The process-wide dictionary lives in a module-level immutable container.
# Reassigning the *binding* below is startup-only; the Dictionary object itself
# has no mutating methods after construction.
class _ProcessDictionary:
  """Holds the one shared :class:`Dictionary` for the process."""

  __slots__ = ("dictionary",)

  def __init__(self):
    self.dictionary: Dictionary | None = None

  def set(self, dictionary: Dictionary | None) -> None:
    self.dictionary = dictionary

  def get(self) -> Dictionary | None:
    return self.dictionary

  def entries(self) -> tuple[bytes, ...]:
    return self.dictionary.entries if self.dictionary is not None else ()


_process_dictionary = _ProcessDictionary()


def set_dictionary(dictionary: Dictionary | None) -> None:
  """Installs the process-wide dictionary.  Called once, during startup.

  Once the registry is frozen (``Setup()`` succeeded) the dictionary can no
  longer change; a later attempt raises :class:`RegistryFrozenError` rather
  than silently splitting the engine and strategy views.
  """
  if _frozen:
    raise RegistryFrozenError(
        f"Cannot replace the dictionary: {_frozen_reason}.")
  _process_dictionary.set(dictionary)


def get_dictionary() -> Dictionary | None:
  """The shared dictionary, or ``None`` when none was provided."""
  return _process_dictionary.get()


def get_dictionary_entries() -> tuple[bytes, ...]:
  """The shared entries, in order (empty when no dictionary was provided)."""
  return _process_dictionary.entries()


def render_dictionary_path(dictionary: Dictionary) -> str:
  """Materializes a dictionary as a libFuzzer ``-dict=`` file.

  The file is a serialization of the parsed :class:`Dictionary`; there is
  still only one in-memory entry list.  The file must outlive ``Setup()``
  because the engine reads it when fuzzing begins, so it is removed at process
  exit rather than here.
  """
  handle, path = tempfile.mkstemp(prefix="atheris-dict-", suffix=".txt")
  try:
    os.close(handle)
    dictionary.write_libfuzzer_file(path)
  except BaseException:
    try:
      os.unlink(path)
    except OSError:
      pass
    raise
  atexit.register(_remove_quietly, path)
  return path


def _remove_quietly(path: str) -> None:
  try:
    os.unlink(path)
  except OSError:
    pass


def DescribeMutationConfiguration() -> str:
  """A human-readable summary of the active mutation setup.

  Printed during ``Setup()`` so logs record which strategy and dictionary were
  active, and whether generic mutations issued from inside the strategy consult
  the dictionary.
  """
  lines = []
  if _active_invoker is not None:
    lines.append(f"INFO: Custom mutator: {_active_invoker.strategy.name!r}")
    if _active_invoker.dictionary_aware_mutate:
      lines.append(
          "INFO: atheris.Mutate() calls made inside the custom mutator use "
          "the dictionary (decision fixed at startup).")
    else:
      lines.append(
          "INFO: atheris.Mutate() calls made inside the custom mutator do not "
          "use a dictionary (no dictionary provided; fixed at startup).")
  else:
    lines.append("INFO: Custom mutator: <none>")

  dictionary = _process_dictionary.get()
  if dictionary is not None:
    lines.append(f"INFO: {dictionary.describe()}")
    lines.append(f"INFO: Dictionary sha256: {dictionary.digest}")
    lines.append("INFO: atheris.Mutate() uses the dictionary.")
  else:
    lines.append("INFO: No dictionary provided.")
  return "\n".join(lines)
