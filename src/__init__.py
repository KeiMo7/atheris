# Copyright 2021 Google Inc.
# Copyright 2021 Fraunhofer FKIE
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
"""Atheris is a coverage-guided Python fuzzing engine."""

import sys
from typing import List, Optional

from .function_hooks import _hook_str
from .function_hooks import enabled_hooks
from .function_hooks import gen_match
from .import_hook import instrument_imports

from .instrument_bytecode import instrument_all
from .instrument_bytecode import instrument_func
from .instrument_bytecode import patch_code

# Named custom mutation strategies, the shared dictionary, and the errors
# raised by both.  ``Setup`` below is a thin Python layer over the native one;
# it resolves strategy names, loads dictionaries, and freezes the registry.
from . import custom_mutators as _custom_mutators
from . import native as _native
from .custom_mutators import AtherisSetupError
from .custom_mutators import CustomMutatorContractError
from .custom_mutators import CustomMutatorError
from .custom_mutators import CustomMutatorOutputTooLongError
from .custom_mutators import CustomMutatorRuntimeError
from .custom_mutators import CustomMutatorTypeError
from .custom_mutators import Dictionary
from .custom_mutators import DictionaryError
from .custom_mutators import MAX_DICTIONARY_ENTRY_SIZE
from .custom_mutators import MAX_DICTIONARY_SIZE
from .custom_mutators import MutationStrategy
from .custom_mutators import RegistryFrozenError
from .custom_mutators import StrategyInvoker
from .custom_mutators import UnknownCustomMutatorError
from .custom_mutators import DescribeMutationConfiguration
from .custom_mutators import get_dictionary
from .custom_mutators import get_dictionary_entries
from .custom_mutators import get_registered
from .custom_mutators import invoke_custom_mutator
from .custom_mutators import load_dictionary
from .custom_mutators import parse_dictionary
from .custom_mutators import parse_dictionary_entry
from .custom_mutators import register
from .custom_mutators import register_custom_mutator
from .custom_mutators import register_custom_mutators
from .custom_mutators import registered_custom_mutators
from .custom_mutators import registered_names
from .custom_mutators import unregister_custom_mutator

# MyPy cannot find native code.
from .native import _reserve_counter  # type: ignore[import]
from .native import _trace_branch  # type: ignore[import]
from .native import _trace_cmp  # type: ignore[import]
from .native import _trace_regex_match  # type: ignore[import]
from .native import ALL_REMAINING  # type: ignore[import]
from .native import build_mode  # type: ignore[import]
from .native import Fuzz  # type: ignore[import]
from .native import FuzzedDataProvider  # type: ignore[import]
from .native import Mutate  # type: ignore[import]
from .native import UpdateCounterArrays  # type: ignore[import]
from .utils import path


def _find_dictionary_argument(args):
  """Finds a libFuzzer ``-dict=FILE`` argument.

  Returns ``(path, spans)`` where ``spans`` lists the ``[start, stop)`` ranges
  of ``args`` occupied by the argument (``-dict FILE`` occupies two entries).
  Raises :class:`AtherisSetupError` for malformed or repeated arguments.
  """
  path = None
  spans = []
  index = 0
  while index < len(args):
    argument = args[index]
    if argument.startswith("-dict=") or argument.startswith("--dict="):
      value = argument.split("=", 1)[1]
      if not value:
        raise AtherisSetupError("The -dict= argument requires a file path.")
      if path is not None:
        raise AtherisSetupError(
            "The dictionary may only be specified once on the command line.")
      path = value
      spans.append((index, index + 1))
    elif argument in ("-dict", "--dict"):
      if index + 1 >= len(args):
        raise AtherisSetupError("The -dict argument requires a file path.")
      if path is not None:
        raise AtherisSetupError(
            "The dictionary may only be specified once on the command line.")
      path = args[index + 1]
      if not path:
        raise AtherisSetupError("The -dict argument requires a file path.")
      spans.append((index, index + 2))
      index += 1
    index += 1
  return path, spans


def _replace_dictionary_argument(args, spans, new_path):
  """Returns ``args`` with the original dictionary argument replaced."""
  replacement = f"-dict={new_path}"
  if not spans:
    return list(args) + [replacement]
  first_start = spans[0][0]
  result = _strip_spans(args, spans)
  result.insert(first_start, replacement)
  return result


def _strip_spans(args, spans):
  """Removes the ``[start, stop)`` ranges in ``spans`` from ``args``."""
  result = []
  index = 0
  while index < len(args):
    span = next((s for s in spans if s[0] == index), None)
    if span is not None:
      index = span[1]
      continue
    result.append(args[index])
    index += 1
  return result


def _find_mutator_argument(args):
  """Finds ``-atheris_custom_mutator=NAME`` (or the two-word form).

  Returns ``(name, spans)``.  The argument is Atheris-specific, so it is
  consumed here rather than passed on to libFuzzer.
  """
  name = None
  spans = []
  index = 0
  prefix = "-atheris_custom_mutator="
  bare = "-atheris_custom_mutator"
  while index < len(args):
    argument = args[index]
    if argument.startswith(prefix):
      value = argument[len(prefix):]
      if not value:
        raise AtherisSetupError(
            "The -atheris_custom_mutator= argument requires a name.")
      if name is not None:
        raise AtherisSetupError(
            "The custom mutator may only be selected once on the command "
            "line.")
      name = value
      spans.append((index, index + 1))
    elif argument == bare:
      if index + 1 >= len(args):
        raise AtherisSetupError(
            "The -atheris_custom_mutator argument requires a name.")
      if name is not None:
        raise AtherisSetupError(
            "The custom mutator may only be selected once on the command "
            "line.")
      name = args[index + 1]
      spans.append((index, index + 2))
      index += 1
    index += 1
  return name, spans


def _resolve_strategy(selector):
  """Turns a registered name, a MutationStrategy, or a callable into one."""
  return _custom_mutators.resolve_strategy_for_setup(selector)


def Setup(args,
          test_one_input,
          internal_libfuzzer: Optional[bool] = None,
          custom_mutator=None,
          custom_crossover=None,
          custom_mutator_name: Optional[str] = None,
          mutator_name: Optional[str] = None,
          dictionary=None,
          dict_path=None,
          **kwargs):
  """Prepares Atheris for fuzzing.

  This wraps the native ``Setup`` and adds named custom mutation strategies and
  dictionary support.  All of the native arguments keep their meaning.

  Args:
    args: process arguments, typically ``sys.argv``.  A libFuzzer-style
      ``-dict=FILE`` argument is parsed by Atheris and its entries are shared
      with the built-in mutation path.
    test_one_input: the fuzzing entry point, taking one ``bytes`` argument.
    internal_libfuzzer: whether to use Atheris' built-in libFuzzer.  ``None``
      lets Atheris decide.
    custom_mutator: a callable ``(data, max_size, seed) -> bytes`` (the legacy
      interface), a registered strategy name, or a :class:`MutationStrategy`.
    custom_crossover: a callable
      ``(data1, data2, max_out_size, seed) -> bytes``.
    custom_mutator_name: the name of a registered strategy to select.  Accepts
      the same values as the string form of ``custom_mutator``; supplied
      separately for readability at call sites that make strategy choice
      configurable.
    mutator_name: alias for ``custom_mutator_name``.
    dictionary: a path to a dictionary file, a sequence of ``bytes``/``str``
      entries, or a pre-parsed :class:`Dictionary`.  Takes precedence over
      ``-dict=`` in ``args``.
    dict_path: alias for ``dictionary``.
    **kwargs: reserved for forward compatibility.

  Raises:
    AtherisSetupError: the dictionary is missing or malformed, no strategy
      matching the selected name exists, or the selected strategy requires a
      dictionary that was not provided.  Startup is aborted and the fuzzing
      loop is never entered.
    RegistryFrozenError: if this is called after startup has already begun.
  """
  del kwargs  # Accepted for forward compatibility with the native signature.

  if _custom_mutators.is_frozen():
    raise RegistryFrozenError(
        "Setup() has already completed; the custom mutator selection and "
        "dictionary are fixed for the life of the process.")

  args = list(args)

  if custom_mutator_name is None:
    custom_mutator_name = mutator_name
  elif mutator_name is not None and mutator_name != custom_mutator_name:
    raise AtherisSetupError(
        "custom_mutator_name and mutator_name disagree: "
        f"{custom_mutator_name!r} != {mutator_name!r}")

  # A ``-atheris_custom_mutator=NAME`` argument lets a pipeline select the
  # strategy without changing the call site.  It is stripped before the
  # arguments reach libFuzzer.
  argument_mutator_name, mutator_spans = _find_mutator_argument(args)
  if argument_mutator_name is not None:
    args = _strip_spans(args, mutator_spans)
    if custom_mutator_name is None:
      custom_mutator_name = argument_mutator_name
    elif custom_mutator_name != argument_mutator_name:
      raise AtherisSetupError(
          "The custom mutator was selected twice with different names: "
          f"{custom_mutator_name!r} from Setup() and "
          f"{argument_mutator_name!r} from the command line.")

  argument_dict_path, dict_spans = _find_dictionary_argument(args)
  dictionary_spec = dictionary if dictionary is not None else dict_path
  if dictionary_spec is None:
    dictionary_spec = argument_dict_path

  loaded_dictionary = None
  if dictionary_spec is not None:
    if isinstance(dictionary_spec, Dictionary):
      loaded_dictionary = dictionary_spec
    elif isinstance(dictionary_spec, (str, bytes)) or hasattr(
        dictionary_spec, "__fspath__"):
      loaded_dictionary = load_dictionary(dictionary_spec)
    elif isinstance(dictionary_spec, (list, tuple, set, frozenset)):
      entries = [
          entry.encode("utf-8") if isinstance(entry, str) else entry
          for entry in dictionary_spec
      ]
      loaded_dictionary = Dictionary(entries)
    else:
      raise TypeError(
          "dictionary must be a path, a sequence of bytes/str entries, or a "
          f"Dictionary, got {type(dictionary_spec).__name__}")

  selector = custom_mutator_name
  selected_by_name = selector is not None
  if selector is None and isinstance(custom_mutator, str):
    selector = custom_mutator
    selected_by_name = True
    custom_mutator = None
  elif selector is not None and custom_mutator is not None:
    raise AtherisSetupError(
        "Select the custom mutator either by name or by callable, not both.")
  strategy = _resolve_strategy(selector)
  if custom_mutator is not None:
    # A MutationStrategy instance is an explicit (new API) choice; a plain
    # callable is the legacy interface, whose behavior must not change.
    strategy = _resolve_strategy(custom_mutator)
    selected_by_name = isinstance(custom_mutator, MutationStrategy)

  if (strategy is not None and strategy.requires_dictionary and
      loaded_dictionary is None):
    raise AtherisSetupError(
        f"Custom mutator {strategy.name!r} requires a dictionary, but none "
        "was provided. Refusing to start: pass dictionary=<path> to Setup() "
        "or -dict=<path> on the command line.")

  _custom_mutators.set_dictionary(loaded_dictionary)
  invoker = None
  if strategy is not None:
    invoker = StrategyInvoker(strategy, loaded_dictionary)
  _custom_mutators.set_active_invoker(invoker)

  if loaded_dictionary is not None:
    # Hand the engine a serialization of the exact entries parsed above, so
    # the built-in mutation path and the custom strategies share one ordered
    # set of words rather than two independently parsed copies.
    serialized_path = _custom_mutators.render_dictionary_path(loaded_dictionary)
    args = _replace_dictionary_argument(args, dict_spans, serialized_path)

  if loaded_dictionary is not None or selected_by_name:
    sys.stderr.write(DescribeMutationConfiguration() + "\n")
    sys.stderr.flush()

  native_kwargs = {}
  if internal_libfuzzer is not None:
    native_kwargs["internal_libfuzzer"] = internal_libfuzzer
  if invoker is not None:
    native_kwargs["custom_mutator"] = invoker
  if custom_crossover is not None:
    native_kwargs["custom_crossover"] = custom_crossover

  # Registration and selection are fixed from this point on.  Freeze only
  # after every validation above has passed, so a rejected configuration
  # leaves the process free to correct it, while a successful startup can never
  # be changed behind the engine's back.
  _custom_mutators.freeze()
  return _native.Setup(args, test_one_input, **native_kwargs)


# Capitalized aliases, matching the module's existing style (Setup/Fuzz/Mutate)
# and making the API easy to find from either convention.
RegisterCustomMutator = register_custom_mutator
RegisterCustomMutators = register_custom_mutators
UnregisterCustomMutator = unregister_custom_mutator
ListCustomMutators = registered_names
RegisteredCustomMutators = registered_custom_mutators
LoadDictionary = load_dictionary
ParseDictionary = parse_dictionary
ParseDictionaryEntry = parse_dictionary_entry
GetDictionary = get_dictionary
GetDictionaryEntries = get_dictionary_entries
InvokeCustomMutator = invoke_custom_mutator
MutationStrategyType = MutationStrategy
CustomMutatorStrategy = MutationStrategy


# PyInstaller Support
# PyInstaller doesn't automatically support lazy imports, which happens because
# we dynamically decide whether to import the with/without_libfuzzer versions of
# the core module. This function tells it where to look for a hook-atheris.py
# file.


def get_hook_dirs() -> List[str]:
  import os  # pylint: disable=g-import-not-at-top
  return [os.path.dirname(__file__)]
