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

import base64
import json
from pathlib import Path
import tempfile
import unittest

from atheris.artifact_io import sha256_bytes
from atheris.corpus import FORMAT_NAME
from atheris.corpus import FORMAT_VERSION
from atheris.corpus import export_corpus
from atheris.corpus import import_corpus


class CorpusArchiveTests(unittest.TestCase):

  def setUp(self):
    self.temp_dir = tempfile.TemporaryDirectory()
    self.root = Path(self.temp_dir.name)

  def tearDown(self):
    self.temp_dir.cleanup()

  def test_round_trip_and_skip_corrupt_entries(self):
    corpus = self.root / "corpus"
    corpus.mkdir()
    (corpus / "one").write_bytes(b"one")
    (corpus / "nested").mkdir()
    (corpus / "nested" / "two").write_bytes(b"two\x00bytes")
    (corpus / "three").write_bytes(b"three")

    archive = self.root / "corpus.jsonl"
    export = export_corpus([corpus], archive)
    self.assertEqual(export.exported, 3)

    lines = archive.read_text().splitlines()
    entries = [json.loads(line) for line in lines[1:]]
    entries_by_name = {entry["name"]: entry for entry in entries}
    corrupted = dict(entries_by_name["one"])
    corrupted["data"] = base64.b64encode(b"not-the-original").decode("ascii")
    traversal = {
        "name": "../escape",
        "size": 4,
        "sha256": sha256_bytes(b"evil"),
        "data": base64.b64encode(b"evil").decode("ascii"),
    }
    archive.write_text(
        json.dumps({
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
        }) + "\n" +
        json.dumps(entries_by_name["nested/two"]) + "\n" +
        "{not json}\n" +
        json.dumps(corrupted) + "\n" +
        json.dumps(traversal) + "\n" +
        json.dumps(entries_by_name["three"]) + "\n" +
        '{"name": "truncated"')

    restored = self.root / "restored"
    report = import_corpus(archive, restored)
    self.assertEqual(report.imported, 2)
    self.assertEqual(report.skipped, 4)
    self.assertEqual((restored / "nested" / "two").read_bytes(), b"two\x00bytes")
    self.assertEqual((restored / "three").read_bytes(), b"three")
    self.assertFalse((self.root / "escape").exists())
    self.assertTrue(report.errors)

  def test_rejects_unknown_archive(self):
    archive = self.root / "bad.jsonl"
    archive.write_text("{}\n")
    with self.assertRaises(ValueError):
      import_corpus(archive, self.root / "restored")


if __name__ == "__main__":
  unittest.main()
