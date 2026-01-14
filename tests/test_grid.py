from state2action_rt.grid import grid_id_to_cell_rect, xy_to_grid_id


def test_xy_to_grid_id_corners():
    roi = (0, 0, 600, 900)
    assert xy_to_grid_id(0, 0, roi, 6, 9) == 0
    assert xy_to_grid_id(599, 899, roi, 6, 9) == 53


def test_grid_id_to_cell_rect_center():
    roi = (0, 0, 600, 900)
    rect = grid_id_to_cell_rect(7, roi, 6, 9)
    assert rect == (100, 100, 200, 200)
