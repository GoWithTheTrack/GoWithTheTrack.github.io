"""
python compile_Scene_Explore.py
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
                    "zhizheng_static_720P_stylized/blue_car_orbit.mp4",
                ],
                [
                    "zhizheng_static_new_720P_stylized/blue_car_spiral.mp4",
                ],                  
                [
                    "zhizheng_static_720P_stylized/eth_spiral.mp4",
                ],
                [
                    "zhizheng_static_720P_stylized/meeting_room_zoom_in_120_fov.mp4",
                ],
                [
                    "zhizheng_static_720P_stylized/vgg_interpolate.mp4",
                ],

                [
                    "zhizheng_static_difficult_stylized/lily_dragon_spiral.mp4"
                ],
                [
                    ("zhizheng_static_difficult_stylized/meeting_room_zoom_in_30_fov.mp4", "wan22_480P_depth")
                ],
                [
                    ("zhizheng_static_difficult_stylized/eth_zoom_out.mp4", "wan22_480P_depth")
                ],
                [
                    "zhizheng_static_new_720P_stylized/nature_1_rotate_360.mp4"
                ],
                [
                    "zhizheng_static_new_720P_stylized/vid2sim_interpolate.mp4"
                ],
                [
                    "zhizheng_static_new_720P_stylized/vid2sim_rotate_360.mp4"
                ],
                [
                    "zhizheng_static_new_720P_stylized/vid2sim_single.mp4"
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

def convert_path(examples):
    c_examples = []
    for i in range(len(examples)):
        c_lis = []
        for j in range(len(examples[i])):
            if isinstance(examples[i][j], tuple):
                model_name, path = examples[i][j][1], examples[i][j][0]
            else:
                model_name = "wan22_720P"
                path = examples[i][j] 
            
            dataset_path = path.split('/')[0]
            sample_name = path.split('/')[1].replace('.mp4', '')
            dataset_path = os.path.join("/root/hf_repo/webpage/eval_data/eval_data/", dataset_path)
            c_lis.append((dataset_path, model_name, sample_name))
        c_examples.append(c_lis)
    return c_examples

good_examples = convert_path(good_example)
########################################################################################

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

def process_track_data(track_path):
    """
    Loads track data and scales it to fit the fixed resolution of 480x832.
    Returns (scaled_tr, final_vis).
    """
    track_data = np.load(track_path, allow_pickle=True).item()
    tr = track_data["uvz"]
    if tr.ndim == 4: tr = tr[0]
    
    vis_mask = track_data["vis"]
    if vis_mask.ndim == 4: vis_mask = vis_mask[0]
    vis_mask_bool = vis_mask.astype(bool)
    if vis_mask_bool.shape[-1] == 1: vis_mask_bool = vis_mask_bool[..., 0]
    
    # FIXED TARGET RESOLUTION
    target_h = 480
    target_w = 832
    
    # Get original track dims or default to same
    orig_H = 480 # track_data.get("H", target_h)
    orig_W = 832 #track_data.get("W", target_w)
    
    # Calculate scale factors to match 480x832
    scale_x = target_w / orig_W
    scale_y = target_h / orig_H
    
    # Scale tracks
    scaled_tr = tr.copy()
    scaled_tr[..., 0] *= scale_x
    scaled_tr[..., 1] *= scale_y
    
    # Check visibility against 480x832
    on_screen = (scaled_tr[..., 0] >= 0) & (scaled_tr[..., 1] >= 0) & \
                (scaled_tr[..., 0] < target_w) & (scaled_tr[..., 1] < target_h)
    
    final_vis = (vis_mask_bool & on_screen)[..., None]
    
    return scaled_tr, final_vis

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

def visualize_mesh(example, output_path):
    """
    Layout: Continuous flow in 4 columns.
    Fixed Resolution for tracks/refs: 480x832.
    Labels: Reference Track X, Reference Image X, Video Track, Generated Video.
    """
    if not example:
        return None

    # --- Step 1: Identify Paths ---
    base_path, model_name, sample_name = example[0]
    log_dir = os.path.join(base_path, "log", sample_name)
    source_video_path = os.path.join(log_dir, "video.mp4")
    track_path = os.path.join(log_dir, "track.npy")
    gen_video_path = os.path.join(base_path, model_name, f"{sample_name}.mp4")

    tmp_dir = "/tmp_visuals2"
    os.makedirs(tmp_dir, exist_ok=True)
    
    if not os.path.exists(source_video_path) or not os.path.exists(track_path):
        print(f"Skipping {sample_name}: Missing source or track file.")
        return None

    # Use Source Video for FPS/Duration, but FORCE Width/Height for visualizer
    fps = 15
    duration = 3.0
    
    props = get_video_properties(source_video_path)
    if props:
        fps, _, _, duration = props
        
    # FORCE RESOLUTION FOR TRACKS
    w, h = 832, 480

    # --- Step 2: Gather References ---
    linear_items = [] 
    
    ref_dir = os.path.join(log_dir, "ref")
    
    if os.path.exists(ref_dir):
        idx = 0
        while True:
            ref_img_path = os.path.join(ref_dir, f"{idx}.png")
            ref_npy_path = os.path.join(ref_dir, f"{idx}.npy")
            
            if not (os.path.exists(ref_img_path) and os.path.exists(ref_npy_path)):
                break 
            
            track_out_name = f"{sample_name}_ref_track_{idx}"
            image_out_name = f"{sample_name}_ref_image_{idx}"
            track_out_path = os.path.join(tmp_dir, f"{track_out_name}.mp4")
            image_out_path = os.path.join(tmp_dir, f"{image_out_name}.mp4")
            
            # 1. Generate Ref Track (Fixed 480x832)
            if not os.path.exists(track_out_path):
                try:
                    tr, final_vis = process_track_data(ref_npy_path) # Uses internal hardcoded 480x832
                    frames = np.zeros((len(tr), h, w, 3), dtype=np.uint8) 
                    vis_ref = Visualizer(save_dir=tmp_dir, pad_value=0, linewidth=2, mode="rainbow", fps=int(fps))
                    vis_ref.visualize(video=frames, tracks=tr, visibility=final_vis, filename=track_out_name, save_video=True)
                except Exception as e:
                    print(f"Error creating ref track {idx}: {e}")
            linear_items.append((f"Reference Track {idx+1}", track_out_path))

            # 2. Generate Ref Image (Resized to 480x832)
            if not os.path.exists(image_out_path):
                try:
                    img = cv2.imread(ref_img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (w, h)) # Resize to 480x832
                        # Use track length for duration
                        tr, _ = process_track_data(ref_npy_path)
                        T_len = len(tr)
                        frames = np.array([img] * T_len, dtype=np.uint8)
                        vis_saver = Visualizer(save_dir=tmp_dir, fps=int(fps))
                        vis_saver.save_video_clip(frames, filename=image_out_name)
                except Exception as e:
                    print(f"Error creating ref image {idx}: {e}")
            linear_items.append((f"Reference Image {idx+1}", image_out_path))
            
            idx += 1

    # --- Step 3: Main Video Track (Fixed 480x832) ---
    track_video_name = f"{sample_name}_track"
    track_video_path = os.path.join(tmp_dir, f"{track_video_name}.mp4")

    if not os.path.exists(track_video_path):
        try:
            tr, final_vis = process_track_data(track_path) # Uses internal hardcoded 480x832
            frames = np.zeros((len(tr), h, w, 3), dtype=np.uint8)
            vis = Visualizer(save_dir=tmp_dir, pad_value=0, linewidth=2, mode="rainbow", fps=int(fps))
            vis.visualize(video=frames, tracks=tr, visibility=final_vis, filename=track_video_name, save_video=True)
        except Exception as e:
            print(f"Error generating main track video: {e}")
            return None
            
    linear_items.append(("Video Track", track_video_path))

    # --- Step 4: Generated Video ---
    if os.path.exists(gen_video_path):
        linear_items.append(("Generated Video", gen_video_path))
    else:
        white_video_path = os.path.join(tmp_dir, "white_spacer.mp4")
        get_white_video(white_video_path, width=w, height=h, duration=duration, fps=fps)
        linear_items.append(("Generated Video (Missing)", white_video_path))

    # --- Step 5: Padding ---
    grid_cols = 4 
    total_items = len(linear_items)
    padding_needed = (grid_cols - (total_items % grid_cols)) % grid_cols
    
    if padding_needed > 0:
        white_video_path = os.path.join(tmp_dir, "white_spacer.mp4")
        get_white_video(white_video_path, width=w, height=h, duration=duration, fps=fps)
        for i in range(padding_needed):
            linear_items.append((" " * (i+1), white_video_path))

    # Construct Dictionary
    grid_inputs = {}
    for label, path in linear_items:
        grid_inputs[label] = path

    # --- Step 6: Stack Videos ---
    try:
        print(f"Stacking {len(grid_inputs)} videos into {output_path} (4 cols)...")
        stack_videos.stack_videos(grid_inputs, output_path, cols=grid_cols)
        return output_path
    except Exception as e:
        print(f"Error in stack_videos: {e}")
        return None

if __name__ == "__main__":
    output_dir = "./videos/Multi-Reference Scene Exploration/"
    os.makedirs(output_dir, exist_ok=True)

    for idx, example in enumerate(good_examples):
        save_path = os.path.join(output_dir, str(idx) + ".mp4")
        result = visualize_mesh(example, save_path)
        
        if result:
            print(f"Successfully created: {result}")
        else:
            print(f"Failed to create example {idx}")