/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "custom_mutator.h"

#include <Python.h>
#include <stddef.h>
#include <stdint.h>

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

#include "macros.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace atheris {

namespace py = pybind11;

namespace {
std::function<py::bytes(py::bytes data, size_t max_size, unsigned int seed)>&
    custom_mutator_global = *new std::function<py::bytes(
        py::bytes data, size_t max_size,
        unsigned int seed)>([](py::bytes data, size_t max_size,
                               unsigned int seed) -> py::bytes {
      // This function should never be called.
      _exit(-1);
    });

// Set once a fatal strategy failure has been reported, so it is only printed
// once even if the engine retries mutations.
bool custom_mutator_error_reported = false;
}  // namespace

NO_SANITIZE
size_t custom_mutator(uint8_t* data, size_t size, size_t max_size,
                      unsigned int seed) {
  // llvm::Optional-like explicit "rejected" signal.  The Python invoker returns
  // an empty byte string when a *recoverable* contract violation (non-bytes
  // result or an oversized result) occurs; the engine treats a zero-length
  // result as "this mutation produced nothing" and simply tries another one.
  //
  // Nothing is copied into `data` until the result has been validated by the
  // Python layer and re-checked here, so an oversized result can never reach
  // the engine's input buffer.
  py::bytes result;
  try {
    result = custom_mutator_global(
        py::bytes(reinterpret_cast<const char*>(data), size), max_size, seed);
  } catch (const py::error_already_set& ex) {
    // A Python exception that escaped the strategy wrapper: the strategy
    // raised, or the wrapper itself failed.  The wrapper has already attached
    // the strategy name and original traceback.  Report it and stop this run
    // instead of letting the exception cross the C boundary (which would
    // abort the process with a signal and no diagnostic).
    if (!custom_mutator_error_reported) {
      custom_mutator_error_reported = true;
      std::cerr << "\n"
                << "=== Fatal custom mutator error: ===" << std::endl;
      py::module traceback = py::module::import("traceback");
      try {
        py::object formatted = traceback.attr("format_exception")(
            ex.type(), ex.value(), ex.trace());
        std::cerr << py::str("").attr("join")(formatted).cast<std::string>()
                  << std::endl;
      } catch (...) {
        std::cerr << ex.what() << std::endl;
      }
      std::cerr << "Stopping the fuzzer: the selected custom mutation "
                   "strategy failed. Other registered strategies are "
                   "unaffected."
                << std::endl;
    }
    // Terminate the fuzzing run.  _exit() is used rather than throwing because
    // libFuzzer is compiled without C++ exception support on some platforms;
    // letting an exception escape here is what used to kill the process with
    // SIGABRT and leave no actionable message.
    _exit(1);
  } catch (const std::exception& ex) {
    if (!custom_mutator_error_reported) {
      custom_mutator_error_reported = true;
      std::cerr << "\n=== Fatal custom mutator error: ===" << std::endl
                << ex.what() << std::endl;
    }
    _exit(1);
  }

  std::string new_data = result;
  if (new_data.size() > max_size) {
    // Defense in depth: the Python layer already rejects oversized results,
    // but never write unvalidated data into the engine buffer.
    std::cerr << "WARNING: custom mutator returned " << new_data.size()
              << " bytes, which exceeds max_size=" << max_size
              << "; the mutation was rejected." << std::endl;
    return 0;
  }
  if (new_data.size() > 0) {
    memcpy(data, new_data.data(), new_data.size());
  }
  return new_data.size();
}

NO_SANITIZE
void _set_custom_mutator(
    const std::function<py::bytes(py::bytes data, size_t max_size,
                                  unsigned int seed)>& custom_mutator_func) {
  custom_mutator_global = custom_mutator_func;
}

extern "C" size_t LLVMFuzzerCustomMutator(uint8_t* data, size_t size,
                                          size_t max_size, unsigned int seed) {
  return custom_mutator(data, size, max_size, seed);
}

}  // namespace atheris
