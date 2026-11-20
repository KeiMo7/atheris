#!/usr/bin/python3

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
"""An example of named custom mutators and a shared dictionary.

This is a Python translation of the compression example from
https://github.com/google/fuzzing/blob/master/docs/structure-aware-fuzzing.md,
extended to register two strategies that can be picked by name and to use a
dictionary that both the strategy and the built-in mutation path can see.

Run it like this:

  # Pick the "compressed" strategy.
  python3 example_fuzzers/named_mutator_example.py \
      -atheris_custom_mutator=compressed

  # Pick the identity strategy; the fuzzer will not find the crash.
  python3 example_fuzzers/named_mutator_example.py \
      -atheris_custom_mutator=identity

  # Replay a fixed sequence with a dictionary and a fixed seed.
  python3 example_fuzzers/named_mutator_example.py \
      -atheris_custom_mutator=compressed \
      -dict=example_fuzzers/named_mutator_example.dict \
      -seed=42
"""

import atheris

with atheris.instrument_imports():
  import sys
  import zlib


def compressed_mutator(data, max_size, seed):
  """Mutates the decompressed payload, then recompresses it."""
  try:
    decompressed = zlib.decompress(data)
  except zlib.error:
    decompressed = b"Hi"
  else:
    decompressed = atheris.Mutate(decompressed, len(decompressed))
  return zlib.compress(decompressed)


def identity_mutator(data, max_size, seed):
  """Leaves the compressed payload opaque to the mutator."""
  return data


@atheris.instrument_func
def TestOneInput(data):
  """The entry point for our fuzzer."""
  try:
    decompressed = zlib.decompress(data)
  except zlib.error:
    return

  if len(decompressed) < 2:
    return

  try:
    if decompressed.decode() == "FU":
      raise RuntimeError("Boom")
  except UnicodeDecodeError:
    pass


if __name__ == "__main__":
  atheris.register_custom_mutator(
      "compressed", compressed_mutator, requires_dictionary=False)
  atheris.register_custom_mutator("identity", identity_mutator)

  # Selection happens once, at startup.  The name may come from the command
  # line (-atheris_custom_mutator=NAME) or from setup_kwargs below.
  atheris.Setup(sys.argv, TestOneInput)
  atheris.Fuzz()
