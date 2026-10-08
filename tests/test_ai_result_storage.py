"""Deletion uses disposable project files, never production data or an API."""
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_ai_chat_selection import Fixtures
from Utils.AIWorkspace import Workspace, read_json, write_json, file_sha
from Utils.AIResultStorage import deletion_preview, delete_result, set_result_hidden


class ResultStorageTests(Fixtures):
    def setUp(self):
        super().setUp()
        self.store = Workspace(self.root)

    def result(self, kind="crops"):
        aid, folder, manifest = self.store.create(kind, "本次处理")
        (folder / "images").mkdir()
        (folder / "images" / "output.png").write_bytes(b"generated fixture")
        write_json(folder / "sources.json", [{"source": self.paths[0], "image": "images/output.png"}])
        self.store.finish(folder, manifest, count=1)
        return aid, folder

    def test_delete_only_owned_files_and_preserve_history_sources_and_other_batches(self):
        aid, folder = self.result()
        other, other_folder = self.result()
        run = self.store.location("runs", "history") / "run.json"
        write_json(run, {"artifact_id": aid})
        transaction = self.store.location("transactions", "undo") / "transaction.json"
        write_json(transaction, {"source": self.paths[0], "status": "committed"})
        # An invalid source index must not direct deletion anywhere else.
        write_json(folder / "sources.json", [{"source": self.paths[0], "image": "../../../../images/small.png"}])
        before = self.hashes()
        other_before = {p.name: file_sha(p) for p in other_folder.rglob("*") if p.is_file()}
        plan = deletion_preview(self.store, aid)
        delete_result(self.store, aid, plan["token"])
        self.assertEqual([p.name for p in folder.iterdir()], ["manifest.json"])
        self.assertEqual(read_json(folder / "manifest.json")["status"], "deleted")
        self.assertEqual(before, self.hashes())
        self.assertTrue(run.exists())
        self.assertTrue(transaction.exists())
        self.assertEqual(other_before, {p.name: file_sha(p) for p in other_folder.rglob("*") if p.is_file()})
        with self.assertRaisesRegex(ValueError, "已删除"):
            self.store.artifact(aid)
        self.store.artifact(other)

    def test_hide_persists_and_unhide_keeps_payload_identical(self):
        aid, folder = self.result("candidates")
        before = file_sha(folder / "sources.json")
        set_result_hidden(self.store, aid, True)
        reopened = Workspace(self.root)
        self.assertTrue(reopened.list_entries()[0]["hidden"])
        reopened.artifact(aid)
        set_result_hidden(reopened, aid, False)
        self.assertFalse(reopened.list_entries()[0]["hidden"])
        self.assertEqual(before, file_sha(folder / "sources.json"))

    def test_changed_files_after_confirmation_require_fresh_preview(self):
        aid, folder = self.result()
        plan = deletion_preview(self.store, aid)
        (folder / "new.txt").write_text("another output", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "确认期间发生变化"):
            delete_result(self.store, aid, plan["token"])
        self.assertTrue((folder / "images/output.png").exists())
        self.assertEqual(read_json(folder / "manifest.json")["status"], "completed")

    def test_partial_deletion_can_be_retried_but_not_opened(self):
        aid, folder = self.result()
        before = self.hashes()
        plan = deletion_preview(self.store, aid)
        original = Path.unlink
        def fail_one(path, *args, **kwargs):
            if path.name == "output.png":
                raise PermissionError("fixture file in use")
            return original(path, *args, **kwargs)
        with patch.object(Path, "unlink", fail_one), self.assertRaises(PermissionError):
            delete_result(self.store, aid, plan["token"])
        self.assertEqual(read_json(folder / "manifest.json")["status"], "delete_failed")
        with self.assertRaisesRegex(ValueError, "清理未完成"):
            self.store.artifact(aid)
        self.assertFalse((folder / "sources.json").exists())
        retry = deletion_preview(self.store, aid)
        self.assertEqual(len(retry["files"]), 1)
        delete_result(self.store, aid, retry["token"])
        self.assertEqual(before, self.hashes())

    def test_running_nonvisual_and_invalid_ids_are_not_deletable(self):
        aid, folder, manifest = self.store.create("crops", "still running")
        with self.assertRaises(ValueError):
            deletion_preview(self.store, aid)
        snapshot, _ = self.result("snapshot")
        for value in (snapshot, "../images", "..", str(self.root / "images")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                deletion_preview(self.store, value)
        self.assertEqual(read_json(folder / "manifest.json"), manifest)

    def test_reparse_points_are_rejected_before_any_deletion(self):
        aid, folder = self.result()
        before = self.hashes()
        original = Path.lstat
        # Windows directory junctions carry FILE_ATTRIBUTE_REPARSE_POINT even
        # when is_symlink() is false. Exercise that case without privileges.
        def junction_info(path, *args, **kwargs):
            if path == folder / "images":
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
            return original(path, *args, **kwargs)
        with patch.object(Path, "lstat", junction_info), self.assertRaisesRegex(ValueError, "目录联接"):
            deletion_preview(self.store, aid)
        self.assertTrue((folder / "images/output.png").exists())
        self.assertEqual(before, self.hashes())

    def test_real_symbolic_link_to_sources_is_never_followed(self):
        aid, folder = self.result()
        link = folder / "linked-source.png"
        try:
            os.symlink(self.root / self.paths[0], link)
        except OSError as exc:
            self.skipTest("Creating symlinks is unavailable: " + str(exc))
        try:
            before = self.hashes()
            with self.assertRaises(ValueError):
                deletion_preview(self.store, aid)
            self.assertEqual(before, self.hashes())
        finally:
            link.unlink()
