from state2action_rt.frame_source import list_image_files


def test_list_image_files_sorted_by_name(tmp_path):
    (tmp_path / "002.jpg").write_text("")
    (tmp_path / "001.png").write_text("")
    (tmp_path / "010.jpeg").write_text("")

    files = list_image_files(str(tmp_path), {".png", ".jpg", ".jpeg"})
    assert [f.split("/")[-1].split("\\")[-1] for f in files] == [
        "001.png",
        "002.jpg",
        "010.jpeg",
    ]
