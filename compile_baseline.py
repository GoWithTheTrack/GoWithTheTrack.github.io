"""
python compile_baseline.py

Now we are comparing against baselines. Here we always have a single refernece frame (correspond to first frame).
Can you display in 3 x 3 grid in the following manner:

Source Video | Video Track | Reference Image (denoted as "First Frame" in title)
Tora | Diffusion as Shader | Go-With-The-Flow 
 ATI | Wan-Move | Generated Video (denoted as "Ours")

Please replace model name  in the path with baselines[i] for obtaining videos correponding to model_names[i] defined below.
If the path does not exist, please pad it with a blank video.
Please follow the title name I specified above.
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
baselines = ["Tora", "DaS","GWTF","ATI","WanMove"]
model_names = ["Tora", "Diffusion as Shader", "Go-With-The-Flow", "ATI", "Wan-Move"]
good_example = [
                [
                    "stylized_davis2017_delta_AblationPureDelta_stylized/bike-packing_3.mp4",
                ],     
                [
                    "stylized_davis2017_delta_AblationPureDelta_stylized/bmx-bumps_2.mp4",
                ],
                [
                    "stylized_davis2017_delta_AblationPureDelta_stylized/bmx-bumps_3.mp4",
                ],
                [
                    "stylized_davis2017_delta_AblationPureDelta_stylized/bmx-trees_0.mp4",
                ],
                [
                    ("stylized_davis2017_delta_AblationPureDelta_stylized/bmx-trees_9.mp4", "wan22_480P"),
                ],
                [
                    "stylized_davis2017_delta_AblationPureDelta_stylized/boxing-fisheye_6.mp4",
                ],
                [
                    "stylized_davis2017_delta_AblationPureDelta_stylized/breakdance-flare_0.mp4",
                ],
                [
                    "stylized_davis2017_delta_AblationPureDelta_stylized/breakdance-flare_8.mp4",
                ],
                [
                    "stylized_davis2017_delta_AblationPureDelta_stylized/breakdance_1.mp4",
                ],
                [
                    "stylized_davis2017_delta_AblationPureDelta_stylized/bus_3.mp4",
                ],
                [
                    "stylized_davis2017_delta_AblationPureDelta_stylized/car-roundabout_2.mp4",
                ],
                [
                    "stylized_davis2017_delta_AblationPureDelta_stylized/cat-girl_4.mp4",
                ],
                [
                    "stylized_davis2017_delta_AblationPureDelta_stylized/color-run_9.mp4",
                ],
                [
                    "stylized_davis2017_delta_AblationPureDelta_stylized/dogs-jump_5.mp4",   
                ],
                [
                    ("stylized_davis2017_delta_AblationPureDelta_stylized/dance-jump_5.mp4",  "wan22_480P"),
                ],
                [
                    ("stylized_davis2017_delta_AblationPureDelta_stylized/dancing_0.mp4",  "wan22_480P"),
                ],
                [
                    ("stylized_davis2017_delta_AblationPureDelta_stylized/drone_7.mp4",  "wan22_480P"),
                ],
                [
                    "stylized_davis2017_delta_AblationPureDelta_stylized/dance-twirl_5.mp4"
                ],   
                [
                    "stylized_davis2017_delta_AblationPureDelta_stylized/disk_jockey_5.mp4"
                ],
                [
                    ("stylized_davis2017_delta_AblationPureDelta_stylized/gold-fish_0.mp4",  "wan22_480P"),
                ],
                [
                    ("stylized_davis2017_delta_AblationPureDelta_stylized/scooter-board_2.mp4", "wan22_480P"),   
                ],
                [
                    ("stylized_davis2017_delta_AblationPureDelta_stylized/libby_2.mp4",  "wan22_480P"),
                ],
                [
                    "stylized_davis2017_delta_AblationPureDelta_stylized/libby_0.mp4",
                ],
                [
                    ("stylized_davis2017_delta_AblationPureDelta_stylized/loading_9.mp4", "wan22_480P")
                ],
                [
                    "stylized_davis2017_delta_AblationPureDelta_stylized/lindy_hop_1.mp4"
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
    
    orig_H = track_data.get("H", 480)
    orig_W = track_data.get("W", 832)
    
    scale_x = target_w / orig_W
    scale_y = target_h / orig_H
    
    scaled_tr = tr.copy()
    scaled_tr[..., 0] *= scale_x
    scaled_tr[..., 1] *= scale_y
    
    on_screen = (scaled_tr[..., 0] >= 0) & (scaled_tr[..., 1] >= 0) & \
                (scaled_tr[..., 0] < target_w) & (scaled_tr[..., 1] < target_h)
    
    final_vis = (vis_mask_bool & on_screen)[..., None]
    
    return scaled_tr, final_vis

def generate_ref_clip(ref_dir, read_name, save_name, type, width, height, fps, duration, tmp_dir):
    """
    Helper to generate reference clips.
    """
    clip_path = os.path.join(tmp_dir, f"{save_name}_{type}.mp4")
    
    if os.path.exists(clip_path):
        return clip_path
        
    img_path = os.path.join(ref_dir, f"{read_name}.png")
    npy_path = os.path.join(ref_dir, f"{read_name}.npy")
    
    if not os.path.exists(img_path):
        return None
        
    if not os.path.exists(npy_path):
        pass

    try:
        if type == "img":
            img = cv2.imread(img_path)
            if img is None: return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (width, height))
            num_frames = int(duration * fps)
            frames = np.array([img] * num_frames, dtype=np.uint8)
            vis = Visualizer(save_dir=tmp_dir, fps=int(fps))
            vis.save_video_clip(frames, filename=f"{save_name}_{type}")
            
        elif type == "trk":
            if not os.path.exists(npy_path): return None
            tr, final_vis = process_track_data(npy_path, height, width)
            frames = np.zeros((len(tr), height, width, 3), dtype=np.uint8)
            vis = Visualizer(save_dir=tmp_dir, pad_value=0, linewidth=2, mode="rainbow", fps=int(fps))
            vis.visualize(video=frames, tracks=tr, visibility=final_vis, filename=f"{save_name}_{type}", save_video=True)
            
        return clip_path
    except Exception as e:
        print(f"Failed to generate ref {type} for {save_name}: {e}")
        return None

def visualize_mesh(example, output_path):
    """
    3x3 Grid Layout (Labels Cleaned):
    Source Video | Video Track | First Frame
    Tora         | Diffusion as Shader | Go-With-The-Flow
    ATI          | Wan-Move            | Ours
    """
    if not example or len(example) == 0:
        return None

    # --- Step 1: Base Info ---
    first_item = example[0]
    base_path, model_name, sample_name = first_item
    
    log_dir_common = os.path.join(base_path, "log", sample_name)
    source_video_path = os.path.join(log_dir_common, "video.mp4")
    
    gen_video_path_ours = os.path.join(base_path, model_name, f"{sample_name}.mp4")
    
    props = None
    if os.path.exists(gen_video_path_ours):
        props = get_video_properties(gen_video_path_ours)
    if not props and os.path.exists(source_video_path):
        props = get_video_properties(source_video_path)
    
    if props:
        fps, w, h, duration = props
    else:
        fps, w, h, duration = 15, 832, 480, 3.0
    
    if w % 2 != 0: w -= 1
    if h % 2 != 0: h -= 1

    tmp_dir = "/tmp_visuals_baseline"
    os.makedirs(tmp_dir, exist_ok=True)
    
    white_video_path = os.path.join(tmp_dir, "white_spacer.mp4")
    get_white_video(white_video_path, width=w, height=h, duration=duration, fps=fps)

    # --- Step 2: Prepare Common Elements ---
    
    track_path = os.path.join(log_dir_common, "track.npy")
    main_trk_name = f"{sample_name}_main_trk"
    main_trk_path = os.path.join(tmp_dir, f"{main_trk_name}.mp4")
    
    if not os.path.exists(main_trk_path):
        try:
            if os.path.exists(track_path):
                tr, final_vis = process_track_data(track_path, h, w)
                frames = np.zeros((len(tr), h, w, 3), dtype=np.uint8)
                vis = Visualizer(save_dir=tmp_dir, pad_value=0, linewidth=2, mode="rainbow", fps=int(fps))
                vis.visualize(video=frames, tracks=tr, visibility=final_vis, filename=main_trk_name, save_video=True)
            else:
                get_white_video(main_trk_path, w, h, duration, fps)
        except:
            get_white_video(main_trk_path, w, h, duration, fps)

    ref_dir = os.path.join(log_dir_common, "ref")
    ref_0_name = f"{sample_name}_ref_0_img"
    ref_0_path = generate_ref_clip(ref_dir, "0", ref_0_name, "img", w, h, fps, duration, tmp_dir)
    if not ref_0_path:
        ref_0_path = white_video_path

    # --- Step 3: Prepare Baseline Videos ---
    baseline_video_paths = {}
    
    for i, bl_key in enumerate(baselines):
        bl_path = os.path.join(base_path, bl_key, f"{sample_name}.mp4")
        
        if os.path.exists(bl_path):
            baseline_video_paths[model_names[i]] = bl_path
        else:
            baseline_video_paths[model_names[i]] = white_video_path

    # --- Step 4: Build 3x3 Grid ---
    # Using list of tuples to maintain order and clean labels
    grid_list = []

    # Row 1
    src_p = source_video_path if os.path.exists(source_video_path) else white_video_path
    grid_list.append(("Source Video", src_p))
    grid_list.append(("Video Track", main_trk_path))
    grid_list.append(("First Frame", ref_0_path))

    # Row 2
    grid_list.append(("Tora", baseline_video_paths.get("Tora", white_video_path)))
    grid_list.append(("Diffusion as Shader", baseline_video_paths.get("Diffusion as Shader", white_video_path)))
    grid_list.append(("Go-With-The-Flow", baseline_video_paths.get("Go-With-The-Flow", white_video_path)))

    # Row 3
    grid_list.append(("ATI", baseline_video_paths.get("ATI", white_video_path)))
    grid_list.append(("Wan-Move", baseline_video_paths.get("Wan-Move", white_video_path)))
    ours_p = gen_video_path_ours if os.path.exists(gen_video_path_ours) else white_video_path
    grid_list.append(("Ours", ours_p))

    # Convert to Dict
    grid_inputs = {label: path for label, path in grid_list}

    # --- Step 5: Stack ---
    try:
        print(f"Stacking {len(grid_inputs)} videos into {output_path}...")
        stack_videos.stack_videos(grid_inputs, output_path, cols=3)
        return output_path
    except Exception as e:
        print(f"Error in stack_videos: {e}")
        return None

if __name__ == "__main__":
    output_dir = "./videos/First Frame Stylization and Baseline Comparison/"
    os.makedirs(output_dir, exist_ok=True)

    for idx, example in enumerate(good_examples):
        save_path = os.path.join(output_dir, str(idx) + ".mp4")
        result = visualize_mesh(example, save_path)
        
        if result:
            print(f"Successfully created: {result}")
        else:
            print(f"Failed to create example {idx}")