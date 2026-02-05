"""
python compile_MultiStyle.py
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
                  ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/10040770_0.mp4", "wan22_480P"),
                  ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/10040770_4.mp4", "wan22_480P"),
                ],
                [
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/10652887_0.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/10652887_1.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/10652887_2.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/10652887_3.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/10652887_6.mp4",  "wan22_480P"),
                ],
                [
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/1_0.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/1_4.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/1_6.mp4", "wan22_480P"),
                ],
                [
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/2035509-hd_1920_1080_24fps_0.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/2035509-hd_1920_1080_24fps_3.mp4", "wan22_480P")
                ],
                [
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/2795749-uhd_3840_2160_25fps_1.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/2795749-uhd_3840_2160_25fps_2.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/2795749-uhd_3840_2160_25fps_3.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/2795749-uhd_3840_2160_25fps_4.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/2795749-uhd_3840_2160_25fps_7.mp4", "wan22_480P")
                ],
                [
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/2865340-uhd_3840_2160_30fps_0.mp4", "wan22_480P_depth"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/2865340-uhd_3840_2160_30fps_2.mp4", "wan22_480P_depth"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/2865340-uhd_3840_2160_30fps_3.mp4", "wan22_480P_depth"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/2865340-uhd_3840_2160_30fps_5.mp4", "wan22_480P_depth"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/2865340-uhd_3840_2160_30fps_6.mp4", "wan22_480P_depth"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/2865340-uhd_3840_2160_30fps_7.mp4", "wan22_480P_depth"),
                ],
                [
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/5192151-hd_1920_1080_30fps_0.mp4", "wan22_480P_depth"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/5192151-hd_1920_1080_30fps_1.mp4", "wan22_480P_depth"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/5192151-hd_1920_1080_30fps_2.mp4", "wan22_480P_depth"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/5192151-hd_1920_1080_30fps_3.mp4", "wan22_480P_depth"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/5192151-hd_1920_1080_30fps_4.mp4", "wan22_480P_depth"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/5192151-hd_1920_1080_30fps_5.mp4", "wan22_480P_depth"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/5192151-hd_1920_1080_30fps_6.mp4", "wan22_480P_depth"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/5192151-hd_1920_1080_30fps_7.mp4", "wan22_480P_depth"),
                ],
                [
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/7520246-hd_1920_1080_24fps_0.mp4", "wan22_480P_depth"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/7520246-hd_1920_1080_24fps_1.mp4", "wan22_480P_depth"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/7520246-hd_1920_1080_24fps_2.mp4", "wan22_480P_depth"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/7520246-hd_1920_1080_24fps_3.mp4", "wan22_480P_depth"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/7520246-hd_1920_1080_24fps_4.mp4", "wan22_480P_depth"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/7520246-hd_1920_1080_24fps_5.mp4", "wan22_480P_depth"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/7520246-hd_1920_1080_24fps_6.mp4", "wan22_480P_depth"),
                ],
                [
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/7678714-uhd_3840_2160_25fps_0.mp4", "wan22_480P_depth"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/7678714-uhd_3840_2160_25fps_1.mp4", "wan22_480P_depth"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/7678714-uhd_3840_2160_25fps_2.mp4", "wan22_480P_depth"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/7678714-uhd_3840_2160_25fps_3.mp4", "wan22_480P_depth"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/7678714-uhd_3840_2160_25fps_4.mp4", "wan22_480P_depth"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/7678714-uhd_3840_2160_25fps_5.mp4", "wan22_480P_depth"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/7678714-uhd_3840_2160_25fps_6.mp4", "wan22_480P_depth"),
                ],
                [
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/9489172-uhd_4096_2160_25fps_0.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/9489172-uhd_4096_2160_25fps_1.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/9489172-uhd_4096_2160_25fps_2.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/9489172-uhd_4096_2160_25fps_3.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/9489172-uhd_4096_2160_25fps_6.mp4", "wan22_480P"),
                ],
                [
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/9631907-uhd_4096_2160_24fps_0.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/9631907-uhd_4096_2160_24fps_1.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/9631907-uhd_4096_2160_24fps_2.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/9631907-uhd_4096_2160_24fps_3.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/9631907-uhd_4096_2160_24fps_4.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/9631907-uhd_4096_2160_24fps_5.mp4", "wan22_480P"),
                ],
                [
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_fitness_0_0.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_fitness_0_1.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_fitness_0_3.mp4", "wan22_480P"),
                    "sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_fitness_0_5.mp4",
                ],
                [
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_fitness_6_0.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_fitness_6_1.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_fitness_6_2.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_fitness_6_3.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_fitness_6_4.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_fitness_6_5.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_fitness_6_6.mp4", "wan22_480P"),
                ],
                [
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_other_transportation_11_0.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_other_transportation_11_1.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_other_transportation_11_2.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_other_transportation_11_3.mp4", "wan22_480P"),("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_other_transportation_11_4.mp4", "wan22_480P")
                ],
                [
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_other_transportation_14_0.mp4", "wan22_480P"),
                     ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_other_transportation_14_1.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_other_transportation_14_2.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_other_transportation_14_3.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_other_transportation_14_4.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_other_transportation_14_5.mp4", "wan22_480P"),       
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_other_transportation_14_6.mp4", "wan22_480P"),       
                ],
                [
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_skateboard_1_0.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_skateboard_1_1.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_skateboard_1_2.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_skateboard_1_3.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_skateboard_1_4.mp4", "wan22_480P"),
                    ("sprite_stylized_CustomVideo_Ablation4Uniform_stylized/Pexels_skateboard_1_5.mp4", "wan22_480P")
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
    
    # Calculate scale factors
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
    Detailed Multi-Style Layout with CLEAN Labels:
    Row 1: Source Video | Video Track | Generated Video | (Pad)
    Row 2: Reference Track 1..4
    Row 3: Reference Image 1..4
    Row 4+: Other Stylization 1..N
    """
    if not example or len(example) == 0:
        return None

    # --- Step 1: Base Info from First Example ---
    first_item = example[0]
    base_path, model_name, sample_name = first_item
    
    log_dir = os.path.join(base_path, "log", sample_name)
    source_video_path = os.path.join(log_dir, "video.mp4")
    track_path = os.path.join(log_dir, "track.npy")
    gen_video_path_1 = os.path.join(base_path, model_name, f"{sample_name}.mp4")

    tmp_dir = "/tmp_visuals_multistyle"
    os.makedirs(tmp_dir, exist_ok=True)

    # Determine props from the first generated video
    props = None
    if os.path.exists(gen_video_path_1):
        props = get_video_properties(gen_video_path_1)
    
    if not props and os.path.exists(source_video_path):
        props = get_video_properties(source_video_path)
        
    if props:
        fps, w, h, duration = props
    else:
        fps, w, h, duration = 15, 832, 480, 3.0
    
    if w % 2 != 0: w -= 1
    if h % 2 != 0: h -= 1

    white_video_path = os.path.join(tmp_dir, "white_spacer.mp4")
    get_white_video(white_video_path, width=w, height=h, duration=duration, fps=fps)

    # We use a list of tuples (Label, Path) to ensure order
    grid_list = []

    # ==========================
    # ROW 1: Source | Track | Gen 1 | Pad
    # ==========================
    
    # 1. Source
    src_path = source_video_path if os.path.exists(source_video_path) else white_video_path
    grid_list.append(("Source Video", src_path))
    
    # 2. Track (Visualized)
    track_video_name = f"{sample_name}_track_viz"
    track_video_path = os.path.join(tmp_dir, f"{track_video_name}.mp4")
    
    if not os.path.exists(track_video_path):
        try:
            if os.path.exists(track_path):
                tr, final_vis = process_track_data(track_path, h, w)
                frames = np.zeros((len(tr), h, w, 3), dtype=np.uint8)
                vis = Visualizer(save_dir=tmp_dir, pad_value=0, linewidth=2, mode="rainbow", fps=int(fps))
                vis.visualize(video=frames, tracks=tr, visibility=final_vis, filename=track_video_name, save_video=True)
            else:
                get_white_video(track_video_path, width=w, height=h, duration=duration, fps=fps)
        except Exception as e:
            print(f"Error gen track: {e}")
            get_white_video(track_video_path, width=w, height=h, duration=duration, fps=fps)
            
    grid_list.append(("Video Track", track_video_path))

    # 3. Gen Video 1
    gen1_path = gen_video_path_1 if os.path.exists(gen_video_path_1) else white_video_path
    grid_list.append(("Generated Video", gen1_path))

    # 4. Padding
    # Use spaces for padding keys so no label text appears
    grid_list.append((" ", white_video_path))

    # ==========================
    # ROW 2 & 3: References
    # ==========================
    
    ref_dir = os.path.join(log_dir, "ref")
    ref_tracks = []
    ref_images = []
    
    if os.path.exists(ref_dir):
        idx = 0
        while True:
            ref_img_path = os.path.join(ref_dir, f"{idx}.png")
            ref_npy_path = os.path.join(ref_dir, f"{idx}.npy")
            
            if not (os.path.exists(ref_img_path) and os.path.exists(ref_npy_path)):
                break
                
            t_out_name = f"{sample_name}_ref_t_{idx}"
            i_out_name = f"{sample_name}_ref_i_{idx}"
            t_path = os.path.join(tmp_dir, f"{t_out_name}.mp4")
            i_path = os.path.join(tmp_dir, f"{i_out_name}.mp4")
            
            # Generate Ref Track
            if not os.path.exists(t_path):
                try:
                    tr, final_vis = process_track_data(ref_npy_path, h, w)
                    frames = np.zeros((len(tr), h, w, 3), dtype=np.uint8)
                    vis = Visualizer(save_dir=tmp_dir, pad_value=0, linewidth=2, mode="rainbow", fps=int(fps))
                    vis.visualize(video=frames, tracks=tr, visibility=final_vis, filename=t_out_name, save_video=True)
                except:
                    get_white_video(t_path, width=w, height=h, duration=duration, fps=fps)
            ref_tracks.append(t_path)
            
            # Generate Ref Image
            if not os.path.exists(i_path):
                try:
                    img = cv2.imread(ref_img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (w, h))
                        tr, _ = process_track_data(ref_npy_path, h, w) 
                        frames = np.array([img] * len(tr), dtype=np.uint8)
                        vis = Visualizer(save_dir=tmp_dir, fps=int(fps))
                        vis.save_video_clip(frames, filename=i_out_name)
                    else:
                        get_white_video(i_path, width=w, height=h, duration=duration, fps=fps)
                except:
                    get_white_video(i_path, width=w, height=h, duration=duration, fps=fps)
            ref_images.append(i_path)
            
            idx += 1

    # Fill Row 2 (Ref Tracks) - Max 4
    for c in range(4):
        if c < len(ref_tracks):
            grid_list.append((f"Reference Track {c+1}", ref_tracks[c]))
        else:
            grid_list.append((" " * (c+2), white_video_path))

    # Fill Row 3 (Ref Images) - Max 4
    for c in range(4):
        if c < len(ref_images):
            grid_list.append((f"Reference Image {c+1}", ref_images[c]))
        else:
            grid_list.append((" " * (c+10), white_video_path))

    # ==========================
    # ROW 4+: Other Stylizations
    # ==========================
    
    other_stylizations = []
    # Collect from index 1 onwards
    for i in range(1, len(example)):
        bp, mn, sn = example[i]
        gp = os.path.join(bp, mn, f"{sn}.mp4")
        if os.path.exists(gp):
            other_stylizations.append(gp)
        else:
            other_stylizations.append(white_video_path) 

    # Pad Others to multiple of 4
    total_others = len(other_stylizations)
    padding_needed = (4 - (total_others % 4)) % 4
    for _ in range(padding_needed):
        other_stylizations.append(white_video_path)
    
    for i, vid_path in enumerate(other_stylizations):
        is_pad = (vid_path == white_video_path)
        # Original index is i+1 relative to the 'other' list
        label_text = f"Other Stylization {i+1}" if not is_pad else " " * (i+20)
        grid_list.append((label_text, vid_path))

    # --- Convert to Dict (Preserving Order) ---
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
    output_dir = "./videos/Multi-Reference Stylization/"
    os.makedirs(output_dir, exist_ok=True)

    for idx, example in enumerate(good_examples):
        save_path = os.path.join(output_dir, str(idx) + ".mp4")
        result = visualize_mesh(example, save_path)
        
        if result:
            print(f"Successfully created: {result}")
        else:
            print(f"Failed to create example {idx}")