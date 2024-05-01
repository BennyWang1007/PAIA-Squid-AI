from ml.constants import COL_NUM, ROW_NUM, TILE_SIZE, ACTIONS

def to_tile(x, y, tile_size):
    return x // tile_size, y // tile_size


print(COL_NUM * ROW_NUM)
def to_tile_map(scene_info: dict) -> list:

    tile_map = [0 for _ in range(COL_NUM * ROW_NUM)]
    if "foods" not in scene_info: return tile_map

    for food in scene_info["foods"]:
        x, y = food["x"], food["y"]
        x, y = to_tile(x, y, TILE_SIZE)
        if x < 0 or y < 0 or x >= COL_NUM or y >= ROW_NUM: continue
        tile_map[y * COL_NUM + x] += food["score"]

    return tile_map
