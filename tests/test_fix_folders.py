from lib.datasets import fix_folder_names


def test_renames_double_spaces(tmp_path):
    (tmp_path / "valid" / "American  Spaniel").mkdir(parents=True)
    (tmp_path / "train" / "Pug").mkdir(parents=True)

    changed = fix_folder_names(tmp_path)

    assert len(changed) == 1
    assert (tmp_path / "valid" / "American Spaniel").is_dir()
    assert not (tmp_path / "valid" / "American  Spaniel").exists()


def test_idempotent(tmp_path):
    (tmp_path / "train" / "Pug").mkdir(parents=True)
    assert fix_folder_names(tmp_path) == []