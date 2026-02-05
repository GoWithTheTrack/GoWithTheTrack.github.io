"""
Input:
./video/
    mesh/    
        0.mp4
        1.mp4
    ...
    {application_name}/

Description:
    Step1: Iterate through each dataset : ./vieo/{application} and create a section
        Step2: Iterate through each video
            Step3: sort videos by name
            step4: For html, display each video (please do not show path name this time)

How to run:
    python compile_html.py --input_folder "./videos" 
    cd ../
    zip -r html.zip html/ -x "*.git*" "*__pycache__*"
"""

import os
import glob
import json
import argparse


import os
import glob
import json
import argparse

# --- HTML Template ---
# NOTE: All CSS and JS braces are doubled {{ }} to escape them for Python's .format()
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Compilation Summary</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #121212;
            color: #e0e0e0;
            margin: 0;
            padding: 0;
            display: flex; /* Flex layout for Sidebar + Main Content */
        }}
        
        /* --- Sidebar: Dataset Selection --- */
        .dataset-nav {{
            position: fixed;
            top: 0;
            left: 0;
            width: 250px; /* Fixed width for sidebar */
            height: 100vh;
            background-color: #181818;
            border-right: 1px solid #333;
            padding: 10px;
            z-index: 1002;
            display: flex;
            flex-direction: column; /* Vertical Stacking */
            gap: 5px;
            overflow-y: auto; /* Scrollable if many datasets */
            box-sizing: border-box;
        }}

        /* Scrollbar styling for sidebar */
        .dataset-nav::-webkit-scrollbar {{
            width: 8px;
        }}
        .dataset-nav::-webkit-scrollbar-thumb {{
            background: #444; 
            border-radius: 4px;
        }}
        .dataset-nav::-webkit-scrollbar-track {{
            background: #181818; 
        }}

        .dataset-btn {{
            background-color: transparent;
            color: #aaa;
            border: 1px solid transparent;
            padding: 10px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            text-align: left;
            transition: all 0.2s;
            word-break: break-word; /* Handle long dataset names */
            line-height: 1.4;
        }}

        .dataset-btn:hover {{
            background-color: #2a2d2e;
            color: #fff;
        }}

        .dataset-btn.active {{
            background-color: #37373d;
            color: #fff;
            border-left: 3px solid #0078d4; /* Accent mark on left */
        }}

        /* --- Main Content Layout --- */
        .main-content {{
            margin-left: 250px; /* Leave space for sidebar */
            width: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}

        h1 {{
            text-align: center;
            color: #fff;
            margin-top: 40px; 
            margin-bottom: 30px;
            font-weight: 300;
            font-size: 24px;
        }}
        
        .container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 60px;
            padding-bottom: 100px;
            width: 100%;
        }}
        
        .card {{
            background-color: #000;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 10px rgba(0,0,0,0.5);
            width: 90%; 
            max-width: 1000px; 
            border: 1px solid #333;
            min-height: 200px;
            display: none; /* Hidden by default */
        }}
        
        .card.visible {{
            display: block;
        }}

        video {{
            width: 100%;
            height: auto;
            display: block;
            background: #000;
        }}
        
        .caption-box {{
            padding: 15px;
            background-color: #1a1a1a;
            border-top: 1px solid #333;
            font-family: 'Courier New', Courier, monospace;
            font-size: 14px;
            line-height: 1.5;
            white-space: pre-wrap;
        }}
        
        .caption-line {{
            display: block;
        }}

        /* --- File Path Styling --- */
        .file-path {{
            padding: 8px 15px;
            background-color: #111;
            border-top: 1px solid #222;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 11px;
            color: #666;
            word-break: break-all;
            user-select: text;
        }}
    </style>
    <script>
        document.addEventListener("DOMContentLoaded", function() {{
            // --- State Management ---
            let activeDataset = ""; 

            // --- Lazy Loading Logic ---
            let observer = new IntersectionObserver((entries, observer) => {{
                entries.forEach(entry => {{
                    if (entry.isIntersecting) {{
                        let video = entry.target;
                        if (video.dataset.src) {{
                            video.src = video.dataset.src;
                            video.load();
                            video.removeAttribute('data-src');
                        }}
                        observer.unobserve(video);
                    }}
                }});
            }}, {{ rootMargin: "200px" }});

            // --- View Update Logic ---
            function updateView() {{
                const datasetBtns = document.querySelectorAll('.dataset-btn');
                const cards = document.querySelectorAll('.card');
                const title = document.querySelector('h1');

                // 1. Update UI Buttons
                datasetBtns.forEach(btn => {{
                    if (btn.dataset.target === activeDataset) {{
                        btn.classList.add('active');
                        btn.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
                    }}
                    else btn.classList.remove('active');
                }});

                // 2. Update Title
                title.innerText = `${{activeDataset}}`;

                // 3. Filter Cards (Only check dataset now)
                cards.forEach(card => {{
                    const video = card.querySelector('video');
                    const belongsToDataset = card.dataset.dataset === activeDataset;

                    if (belongsToDataset) {{
                        card.classList.add('visible');
                        if (video.classList.contains('lazy')) {{
                            observer.observe(video);
                        }}
                    }} else {{
                        card.classList.remove('visible');
                        if (!video.paused) {{
                            video.pause();
                        }}
                    }}
                }});
                
                // Reset scroll of main window
                window.scrollTo(0, 0);
            }}

            // --- Event Listeners ---
            
            // Dataset Clicks
            document.querySelectorAll('.dataset-btn').forEach(btn => {{
                btn.addEventListener('click', () => {{
                    activeDataset = btn.dataset.target;
                    updateView();
                }});
            }});

            // --- Initialization ---
            // Select the first dataset button available
            const firstDatasetBtn = document.querySelector('.dataset-btn');
            if (firstDatasetBtn) {{
                activeDataset = firstDatasetBtn.dataset.target;
            }}
            updateView();
        }});
    </script>
</head>
<body>

    <div class="dataset-nav">
        {dataset_nav_buttons}
    </div>

    <div class="main-content">
        <h1>Dataset: Loading...</h1>
        
        <div class="container">
            {video_cards}
        </div>
    </div>

</body>
</html>
"""

def generate_single_index(input_folder):
    # Description Step 1: Scan for datasets
    if not os.path.exists(input_folder):
        print(f"Error: Input directory '{input_folder}' not found.")
        return

    # Sort datasets to ensure tab order is consistent
    datasets = sorted([d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))])
    
    if not datasets:
        print(f"No datasets found inside {input_folder}")
        return

    dataset_nav_html = ""
    all_cards_html = ""

    # Description Step 2: Iterate through each dataset to build tabs and content
    for dataset_name in datasets:
        print(f"Processing dataset: {dataset_name}...")
        
        # 1. Add Navigation Tab for this dataset (Sidebar Button)
        dataset_nav_html += f'<button class="dataset-btn" data-target="{dataset_name}">{dataset_name}</button>\n'

        dataset_path = os.path.join(input_folder, dataset_name)
        mp4_files = sorted(glob.glob(os.path.join(dataset_path, "*.mp4")))

        if not mp4_files:
            continue

        for mp4_path in mp4_files:
            filename = os.path.basename(mp4_path)
            
            # --- PATH LOGIC ---
            # 1. Relative path for the <video src="..."> (so browser can load it)
            relative_video_path = os.path.join(input_folder, dataset_name, filename)
            
            # JSON logic (Optional Caption)
            json_filename = filename.replace(".mp4", ".json")
            json_path = os.path.join(dataset_path, json_filename)
            
            caption_html = ""
            if os.path.exists(json_path):
                caption_html += '<div class="caption-box">'
                try:
                    with open(json_path, 'r') as jf:
                        data = json.load(jf)
                        if isinstance(data, list):
                            for line in data:
                                text = line.get('text', '')
                                color = line.get('color', [255, 255, 255])
                                hex_color = "#{:02x}{:02x}{:02x}".format(*color)
                                caption_html += f'<span class="caption-line" style="color: {hex_color};">{text}</span>'
                except Exception as e:
                    print(f"  Warning: Error reading JSON {json_filename}: {e}")
                caption_html += '</div>'

            # 2. Build Card HTML
            # Removed file path display per requirement
            card_html = f"""
        <div class="card" data-dataset="{dataset_name}">
            <video class="lazy" data-src="{relative_video_path}" preload="none" controls autoplay loop muted playsinline></video>
            {caption_html}
        </div>"""
            all_cards_html += card_html

    # Description Step 3: Write Single Index HTML
    final_html = HTML_TEMPLATE.format(
        dataset_nav_buttons=dataset_nav_html,
        video_cards=all_cards_html
    )

    output_path = "./index.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_html)
    
    print(f"\nSuccessfully generated dashboard at: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile all datasets into a single HTML dashboard.")
    parser.add_argument("--input_folder", type=str, default="./video", help="Path to the folder containing dataset subdirectories.")
    
    args = parser.parse_args()
    
    generate_single_index(args.input_folder)