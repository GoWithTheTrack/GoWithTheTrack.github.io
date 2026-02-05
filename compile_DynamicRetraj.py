"""
python compile_DynamicRetraj.py

Can you change the code so it outputs:
| Source Video | (blank) | Video Track 1 (for example 1) |  Generated Video 1 |
| Video Track 2 (for example 2) | Generated Video 2 | Video Track 3 | Generated Video 3|
...
Please load Video track (track.npy) that is stored under every example folder.
Also Please make sure that the maximum number of columns are capped by 4.
For the title, make sure it does not contain underscores, and Also say Video Track 1 -> Retargeted Track 1, Generated Video 1 -> Retargeted Video 1
"""
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import stack_videos

# Handle MoviePy for generating the white spacer video if needed
try:
    from moviepy.editor import ColorClip
except ImportError:
    from moviepy.video.VideoClip import ColorClip

################### Do not modify ###############################
good_example = [
                [
                  "CustomVideo_DynamicRetrajRelative_Ref4New_Boundary_stylized/2865340-uhd_3840_2160_30fps_RotRight.mp4",
                ],
                [
                    ("CustomVideo_DynamicRetrajRelative_Ref4New_Boundary_stylized/Pexels_singing_3_Backward.mp4", "wan22_480P")
                ],
                [
                    "CustomVideo_DynamicRetrajRelative_Ref4New_Boundary_stylized/Pexels_swimming_9_Backward.mp4"
                ],
                [
                    "CustomVideo_DynamicRetrajRelative_Ref4New_Boundary_stylized/Pexels_winter_sports_6_Backward.mp4"
                ],
                [
                    "CustomVideo_DynamicRetrajRelative_Ref4New_Boundary_stylized/Pexels_yoga_3_spiral1.mp4"
                ],
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/bear_HeroReveal.mp4",
                    ("davis2017_DynamicRetrajRelative_Ref4New_stylized/bear_ArcLeft.mp4", "wan22_480P_depth"),
                    ("davis2017_DynamicRetrajRelative_Ref4New_stylized/bear_Backward.mp4", "wan22_480P_depth"),
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/bear_spiral1.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/bear_RotRight.mp4",
                ],
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/bike-packing_UpForward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/bike-packing_ArcLeft.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/bike-packing_Backward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/bike-packing_HeroReveal.mp4",
                    ("davis2017_DynamicRetrajRelative_Ref4New_stylized/bike-packing_RotRight.mp4", "wan22_480P_depth"),
                    ("davis2017_DynamicRetrajRelative_Ref4New_stylized/bike-packing_spiral1.mp4", "wan22_480P_depth"),
                ],
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/bmx-bumps_spiral1.mp4",
                    ("davis2017_DynamicRetrajRelative_Ref4New_stylized/bmx-bumps_ArcLeft.mp4", "wan22_480P_depth"),
                    ("davis2017_DynamicRetrajRelative_Ref4New_stylized/bmx-bumps_Backward.mp4", "wan22_480P"),
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/bmx-bumps_HeroReveal.mp4",
                    "avis2017_DynamicRetrajRelative_Ref4New_stylized/bmx-bumps_RotRight.mp4"
                ],
                [
                    ("davis2017_DynamicRetrajRelative_Ref4New_stylized/bmx-trees_ArcLeft.mp4", "wan22_480P_depth"),
                    ("davis2017_DynamicRetrajRelative_Ref4New_stylized/bmx-trees_HeroReveal.mp4", "wan22_480P_depth"),
                    ("davis2017_DynamicRetrajRelative_Ref4New_stylized/bmx-trees_UpForward.mp4", "wan22_480P_depth"),
                    
                ],
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/boat_ArcLeft.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/boat_Backward.mp4",
                    #"davis2017_DynamicRetrajRelative_Ref4New_stylized/boat_HeroReveal.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/boat_UpForward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/boat_spiral1.mp4"
                ],  
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/bus_RotRight.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/bus_Backward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/bus_HeroReveal.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/bus_spiral1.mp4",
                ],
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/camel_Backward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/camel_HeroReveal.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/camel_UpForward.mp4",
                    ("davis2017_DynamicRetrajRelative_Ref4New_stylized/camel_spiral1.mp4", "wan22_480P_depth")
                ],
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/car-roundabout_Backward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/car-roundabout_HeroReveal.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/car-roundabout_RotRight.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/car-roundabout_UpForward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/car-roundabout_spiral1.mp4",
                ],
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/car-turn_ArcLeft.mp4",
                    ("davis2017_DynamicRetrajRelative_Ref4New_stylized/car-turn_Backward.mp4", "wan21_720P"),
                    ("davis2017_DynamicRetrajRelative_Ref4New_stylized/car-turn_HeroReveal.mp4", "wan22_480P"),
                    ("davis2017_DynamicRetrajRelative_Ref4New_stylized/car-turn_RotRight.mp4", "wan22_480P_depth")
                ],
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/cat-girl_ArcLeft.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/cat-girl_Backward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/cat-girl_HeroReveal.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/cat-girl_RotRight.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/cat-girl_spiral1.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/cat-girl_UpForward.mp4"
                ],
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/dog_HeroReveal.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/dog_RotRight.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/dog_ArcLeft.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/dog_Backward.mp4",
                    ("davis2017_DynamicRetrajRelative_Ref4New_stylized/dog_spiral1.mp4", "wan22_480P"),
                ],
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/dogs-scale_HeroReveal.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/dogs-scale_RotRight.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/dogs-scale_Backward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/dogs-scale_HeroReveal.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/dogs-scale_spiral1.mp4",
                ],
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/drift-chicane_ArcLeft.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/drift-chicane_spiral1.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/drift-chicane_Backward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/drift-chicane_HeroReveal.mp4",
                ],
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/drift-straight_HeroReveal.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/drift-straight_RotRight.mp4",
                    #"davis2017_DynamicRetrajRelative_Ref4New_stylized/drift-straight_ArcLeft.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/drift-straight_Backward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/drift-straight_UpForward.mp4",
                ],
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/drift-turn_spiral1.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/drift-turn_ArcLeft.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/drift-turn_Backward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/drift-turn_HeroReveal.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/drift-turn_UpForward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/drift-turn_RotRight.mp4"
                ],
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/flamingo_RotRight.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/flamingo_Backward.mp4",
                    ("davis2017_DynamicRetrajRelative_Ref4New_stylized/flamingo_HeroReveal.mp4", "wan22_480P_depth"),
                    ("davis2017_DynamicRetrajRelative_Ref4New_stylized/flamingo_UpForward.mp4", "wan22_480P_depth"),
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/flamingo_spiral1.mp4"
                    
                ],
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/gold-fish_ArcLeft.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/gold-fish_HeroReveal.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/gold-fish_UpForward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/gold-fish_spiral1.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/gold-fish_RotRight.mp4",
                ],
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/hike_ArcLeft.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/hike_Backward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/hike_HeroReveal.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/hike_RotRight.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/hike_UpForward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/hike_spiral1.mp4",
                ],
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/horsejump-high_Backward.mp4",
                    ("davis2017_DynamicRetrajRelative_Ref4New_stylized/horsejump-high_HeroReveal.mp4", "wan22_480P_depth"),
                ],
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/horsejump-low_Backward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/horsejump-low_HeroReveal.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/horsejump-low_RotRight.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/horsejump-low_spiral1.mp4"
                ],
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/koala_spiral1.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/koala_ArcLeft.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/koala_Backward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/koala_HeroReveal.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/koala_RotRight.mp4"
                ],
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/libby_ArcLeft.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/libby_RotRight.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/libby_Backward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/libby_HeroReveal.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/libby_UpForward.mp4"
                ],
                [
                    #"davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/loading_Backward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/loading_ArcLeft.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/loading_Backward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/loading_HeroReveal.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/loading_RotRight.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/loading_spiral1.mp4"
                ],
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/longboard_ArcLeft.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/longboard_HeroReveal.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/longboard_spiral1.mp4"
                ],
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/mbike-trick_ArcLeft.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/mbike-trick_HeroReveal.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/mbike-trick_Backward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/mbike-trick_HeroReveal.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/mbike-trick_RotRight.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/mbike-trick_UpForward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/mbike-trick_spiral1.mp4"
                ],
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/motocross-bumps_ArcLeft.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/motocross-bumps_Backward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/motocross-bumps_HeroReveal.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/motocross-bumps_RotRight.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/motocross-bumps_UpForward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/motocross-bumps_spiral1.mp4",
                ],
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/scooter-gray_HeroReveal.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/scooter-gray_ArcLeft.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/scooter-gray_Backward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/scooter-gray_RotRight.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/scooter-gray_UpForward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/scooter-gray_spiral1.mp4"
                ],
                [
                    "davis2017_DynamicRetrajRelative_Ref4New_Boundary_stylized/soapbox_UpForward.mp4",
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/soapbox_ArcLeft.mp4",
                    ("davis2017_DynamicRetrajRelative_Ref4New_stylized/soapbox_Backward.mp4","wan22_480P"),
                    ("davis2017_DynamicRetrajRelative_Ref4New_stylized/soapbox_HeroReveal.mp4", "wan22_480P"),
                    ("davis2017_DynamicRetrajRelative_Ref4New_stylized/soapbox_RotRight.mp4","wan22_480P"),
                    "davis2017_DynamicRetrajRelative_Ref4New_stylized/soapbox_spiral1.mp4",
                ],                  
            ]

def convert_path(examples):
    c_examples = []
    # Fixed: 'in len()' causes a TypeError, changed to 'in range(len())'
    for i in range(len(examples)):
        c_lis = []
        # Fixed: 'in len()' changed to 'in range(len())'
        for j in range(len(examples[i])):
            
            # Check if the element is a tuple (contains specific model name)
            if isinstance(examples[i][j], tuple):
                model_name, path = examples[i][j][1], examples[i][j][0]
            else:
                model_name = "wan22_720P"
                # Fixed: examples[i][j] is the string itself. [0] would only get the first letter.
                path = examples[i][j] 
            
            # Extract the first directory (e.g., stylized_mesh_zhizheng_stylized)
            dataset_path = path.split('/')[0]
            
            # Extract filename and drop .mp4 (e.g., dancing_1_0)
            sample_name = path.split('/')[1].replace('.mp4', '')
            dataset_path = os.path.join("/root/hf_repo/webpage/eval_data/eval_data/", dataset_path)
            c_lis.append((dataset_path, model_name, sample_name))
        c_examples.append(c_lis)
    return c_examples

good_examples = convert_path(good_example)
########################################################################################

# Visualizer borrowed from reference code
try:
    from moviepy.editor import ImageSequenceClip
except ImportError:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

class Visualizer:
    def __init__(
        self,
        save_dir: str = "./results",
        pad_value: int = 0,
        fps: int = 10,
        mode: str = "rainbow",
        linewidth: int = 1,
        tracks_leave_trace: int = 0,
    ):
        self.mode = mode
        self.save_dir = save_dir
        if mode == "rainbow":
            self.color_map = cm.get_cmap("gist_rainbow")
        elif mode == "cool":
            self.color_map = cm.get_cmap(mode)
            
        self.tracks_leave_trace = tracks_leave_trace
        self.pad_value = pad_value
        self.linewidth = linewidth
        self.fps = fps

    def visualize(self, video: np.ndarray, tracks: np.ndarray, visibility: np.ndarray = None, filename: str = "video", save_video: bool = True):
        # video is (T, H, W, C). Pad H and W.
        if self.pad_value > 0:
            pad_width = ((0,0), (self.pad_value, self.pad_value), (self.pad_value, self.pad_value), (0,0))
            video = np.pad(video, pad_width, mode='constant', constant_values=255)
            tracks = tracks + self.pad_value

        tracking_video = self.draw_tracks_on_video(video=video, tracks=tracks, visibility=visibility, filename=filename)

        if save_video:
            self.save_video_clip(tracking_video, filename=filename, savedir=self.save_dir)
            
        return tracking_video

    def save_video_clip(self, video, filename, savedir=None):
        if savedir is None:
            save_path = os.path.join(self.save_dir, f"{filename}.mp4")
        else:
            save_path = os.path.join(savedir, f"{filename}.mp4")
            
        if isinstance(video, np.ndarray):
            video_list = list(video)
        else:
            video_list = video

        try:
            clip = ImageSequenceClip(video_list, fps=self.fps)
            clip.write_videofile(save_path, codec="libx264", fps=self.fps, logger=None)
        except Exception:
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip as ISC
            clip = ISC(video_list, fps=self.fps)
            clip.write_videofile(save_path, codec="libx264", fps=self.fps, logger=None)

    def draw_tracks_on_video(self, video, tracks, visibility=None, filename=""):
        T, H, W, C = video.shape
        _, N, D = tracks.shape
        
        res_video = [frame.copy().astype(np.uint8) for frame in video]
        vector_colors = np.zeros((T, N, 3))

        if self.mode == "rainbow":
            x_min, x_max = 1e9, -1e9
            y_min, y_max = 1e9, -1e9
            
            for num_tracks in range(N):
                if visibility is not None:
                    vis_col = visibility[:, num_tracks, 0]
                    if np.any(vis_col != 0):
                        f = np.argmax(vis_col != 0)
                    else:
                        f = 0
                else:
                    f = 0
                
                x_min = min(tracks[f, num_tracks, 0], x_min)
                x_max = max(tracks[f, num_tracks, 0], x_max)
                y_min = min(tracks[f, num_tracks, 1], y_min)
                y_max = max(tracks[f, num_tracks, 1], y_max)

            safe_depth = tracks[0, :, 2].copy()
            safe_depth[safe_depth == 0] = 1.0 
            z_inv = 1.0 / safe_depth
            z_min, z_max = np.percentile(z_inv, [2, 98])
            
            norm_x = plt.Normalize(x_min, x_max)
            norm_y = plt.Normalize(y_min, y_max)
            norm_z = plt.Normalize(z_min, z_max)

            for n in range(N):
                if visibility is not None:
                    if np.any(visibility[:, n, 0] != 0):
                        f = np.argmax(visibility[:, n, 0] != 0)
                    else:
                        f = 0
                else:
                    f = 0
                
                r = norm_x(tracks[f, n, 0])
                g = norm_y(tracks[f, n, 1])
                d_val = tracks[0, n, 2] if tracks[0, n, 2] != 0 else 1.0
                b = norm_z(1.0 / d_val)
                
                color = np.array([r, g, b])[None] * 255
                vector_colors[:, n] = np.repeat(color, T, axis=0)

        for t in tqdm(range(T), desc=f"Drawing tracks {filename}", leave=False):
            points_info = []
            for i in range(N):
                coord = (tracks[t, i, 0], tracks[t, i, 1])
                depth = tracks[t, i, 2]
                is_visible = True
                if visibility is not None:
                    is_visible = visibility[t, i, 0] > 0
                
                if coord[0] != 0 and coord[1] != 0:
                      points_info.append((i, coord, depth, is_visible))
            
            points_info.sort(key=lambda x: x[2], reverse=True)
            
            for i, coord, _, is_visible in points_info:
                if is_visible:
                    cv2.circle(res_video[t], (int(coord[0]), int(coord[1])), int(self.linewidth * 2), vector_colors[t, i].tolist(), thickness=-1)

        return np.stack(res_video)

def get_white_video(output_path, width=832, height=480, duration=1.0, fps=15):
    """Generates a white video to be used as padding."""
    if os.path.exists(output_path):
        return output_path
    
    try:
        clip = ColorClip(size=(width, height), color=(255, 255, 255), duration=duration)
        clip.write_videofile(output_path, codec="libx264", fps=fps, audio=False, logger=None)
        return output_path
    except Exception as e:
        print(f"Failed to create white video: {e}")
        return None

def get_video_properties(video_path):
    """Extracts FPS, Width, Height, Duration from a video file."""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 2.0
        cap.release()
        return fps, w, h, duration
    except:
        return None

def process_track_data(track_path, target_h, target_w):
    """
    Loads track data and scales it to match target_h and target_w.
    """
    if not os.path.exists(track_path):
        raise FileNotFoundError(f"Track file not found: {track_path}")

    track_data = np.load(track_path, allow_pickle=True).item()
    tr = track_data["uvz"]
    if tr.ndim == 4: tr = tr[0]
    
    vis_mask = track_data["vis"]
    if vis_mask.ndim == 4: vis_mask = vis_mask[0]
    vis_mask_bool = vis_mask.astype(bool)
    if vis_mask_bool.shape[-1] == 1: vis_mask_bool = vis_mask_bool[..., 0]
    
    # Original track dimensions
    orig_H = track_data.get("H", 480)
    orig_W = track_data.get("W", 832)
    
    # Calculate scale factors to match TARGET video resolution
    scale_x = target_w / orig_W
    scale_y = target_h / orig_H
    
    # Scale tracks
    scaled_tr = tr.copy()
    scaled_tr[..., 0] *= scale_x
    scaled_tr[..., 1] *= scale_y
    
    # Check visibility against TARGET dimensions
    on_screen = (scaled_tr[..., 0] >= 0) & (scaled_tr[..., 1] >= 0) & \
                (scaled_tr[..., 0] < target_w) & (scaled_tr[..., 1] < target_h)
    
    final_vis = (vis_mask_bool & on_screen)[..., None]
    
    return scaled_tr, final_vis

def visualize_mesh(example, output_path):
    """
    Layout: | Source Video | (Blank) | Retargeted Track 1 | Retargeted Video 1 |
            | Retargeted Track 2 | Retargeted Video 2 | Retargeted Track 3 | Retargeted Video 3 | ...
    """
    if not example:
        return None

    # --- Step 1: Base Info from First Example ---
    base_path, _, sample_name = example[0]
    log_dir = os.path.join(base_path, "log", sample_name)
    source_video_path = os.path.join(log_dir, "video.mp4")
    
    # Determine canvas props from FIRST generated video found
    fps, w, h, duration = 15, 832, 480, 3.0
    for item in example:
        bp, mn, sn = item
        gp = os.path.join(bp, mn, f"{sn}.mp4")
        if os.path.exists(gp):
            props = get_video_properties(gp)
            if props:
                fps, w, h, duration = props
                break
    
    if w % 2 != 0: w -= 1
    if h % 2 != 0: h -= 1

    tmp_dir = "/tmp_visuals_Dyn"
    os.makedirs(tmp_dir, exist_ok=True)
    
    white_video_path = os.path.join(tmp_dir, "white_spacer.mp4")
    get_white_video(white_video_path, width=w, height=h, duration=duration, fps=fps)

    # --- Step 2: Build Grid ---
    grid_list = []

    # Row 1 Start: Source | Blank
    src_p = source_video_path if os.path.exists(source_video_path) else white_video_path
    grid_list.append(("Source Video", src_p))
    grid_list.append((" ", white_video_path))

    # Iterate through ALL generated videos
    for idx, item in enumerate(example):
        base_path, model_name, sample_name = item
        
        # Paths
        gen_path = os.path.join(base_path, model_name, f"{sample_name}.mp4")
        log_dir = os.path.join(base_path, "log", sample_name)
        track_path = os.path.join(log_dir, "track.npy")
        
        # 1. Gen Video
        final_gen_path = gen_path if os.path.exists(gen_path) else white_video_path
        if not os.path.exists(gen_path):
            print(f"Warning: Missing video {gen_path}")

        # 2. Specific Track
        track_video_name = f"{sample_name}_track_viz"
        track_video_path = os.path.join(tmp_dir, f"{track_video_name}.mp4")
        
        if not os.path.exists(track_video_path):
            try:
                tr, final_vis = process_track_data(track_path, h, w)
                frames = np.zeros((len(tr), h, w, 3), dtype=np.uint8)
                vis = Visualizer(save_dir=tmp_dir, pad_value=0, linewidth=2, mode="rainbow", fps=int(fps))
                vis.visualize(video=frames, tracks=tr, visibility=final_vis, filename=track_video_name, save_video=True)
            except Exception as e:
                print(f"Error generating track for {sample_name}: {e}")
                get_white_video(track_video_path, width=w, height=h, duration=duration, fps=fps)

        # Append to grid pair: Track | Video
        grid_list.append((f"Retargeted Track {idx+1}", track_video_path))
        grid_list.append((f"Retargeted Video {idx+1}", final_gen_path))

    # --- Convert to Dict ---
    grid_inputs = {label: path for label, path in grid_list}

    # --- Step 5: Stack ---
    try:
        print(f"Stacking {len(grid_inputs)} videos into {output_path}...")
        stack_videos.stack_videos(grid_inputs, output_path, cols=4)
        return output_path
    except Exception as e:
        print(f"Error in stack_videos: {e}")
        return None

if __name__ == "__main__":
    output_dir = "./videos/Camera Retargeting/"
    os.makedirs(output_dir, exist_ok=True)

    for idx, example in enumerate(good_examples):
        save_path = os.path.join(output_dir, str(idx) + ".mp4")
        result = visualize_mesh(example, save_path)
        
        if result:
            print(f"Successfully created: {result}")
        else:
            print(f"Failed to create example {idx}")