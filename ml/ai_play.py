from stable_baselines3 import PPO

from ml.constants import *
from ml.utils import to_tile_map, to_tile

def dfs_best_path(tilemap, x, y, visited, cur_depth, max_depth, cur_score, best_score, cur_path, best_path):

    if x < 0 or x >= COL_NUM or y < 0 or y >= ROW_NUM or visited[x][y] or cur_depth > max_depth:
        return best_score, best_path
    
    visited[x][y] = True
    cur_score += tilemap[y * COL_NUM + x]
    mean_score = 0
    if cur_depth != 0:
        mean_score = cur_score / (cur_depth)
    cur_path.append([x, y])

    if mean_score >= best_score:
        best_score = mean_score
        best_path = cur_path.copy()

    
    best_score, best_path = dfs_best_path(tilemap, x - 1, y, visited, cur_depth + 1, max_depth, cur_score, best_score, cur_path, best_path)
    best_score, best_path = dfs_best_path(tilemap, x + 1, y, visited, cur_depth + 1, max_depth, cur_score, best_score, cur_path, best_path)
    best_score, best_path = dfs_best_path(tilemap, x, y - 1, visited, cur_depth + 1, max_depth, cur_score, best_score, cur_path, best_path)
    best_score, best_path = dfs_best_path(tilemap, x, y + 1, visited, cur_depth + 1, max_depth, cur_score, best_score, cur_path, best_path)

    visited[x][y] = False
    cur_path.pop()

    return best_score, best_path



class MLPlay:

    def __init__(self,*args, **kwargs):
        print("Initial ml script")
        # model1
        self.ppo = PPO.load(R'E:\ppo_model\ppo_swimming_squid_10x10_70_target_fixed2_512000step.zip')


    def update(self, scene_info: dict, *args, **kwargs):
        """
        Generate the command according to the received scene information
        """
        
        state = self.get_state(scene_info)
        # action = self.dqn.predict(state)
        action, _ = self.ppo.predict(state)
        action = ACTIONS[action]

        return [action]


    def get_state(self, scene_info: dict) -> list:

        tilemap = to_tile_map(scene_info)

        if "squid_x" not in scene_info: return [0, 0, 0, 0] + tilemap

        x, y = scene_info["squid_x"], scene_info["squid_y"]
        tile_x, tile_y = to_tile(x, y, TILE_SIZE)
        tx, ty = self.get_target_loc(tile_x, tile_y, tilemap)
        self.tx , self.ty = tx * TILE_SIZE, ty * TILE_SIZE

        # reserve the sign of each value in map, 0 -> 0, 1 -> 1, -1 -> -1
        tilemap = [1 if i > 0 else -1 if i < 0 else 0 for i in tilemap]
        return [tile_x, tile_y, tx - tile_x, ty - tile_y] + tilemap


    def get_target_loc(self, tile_x, tile_y, tilemap) -> tuple[int, int]:

        best_score, best_path = dfs_best_path(tilemap, tile_x, tile_y, [[False for _ in range(ROW_NUM)] for _ in range(COL_NUM)], 0, 7, 0, 0, [], [])

        if len(best_path) == 0 or len(best_path) == 1:
            return tile_x, tile_y
        
        return best_path[1]


    def reset(self):
        """
        Reset the status
        """
        pass


