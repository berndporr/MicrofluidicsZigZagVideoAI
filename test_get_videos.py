import time
from video_processor import get_videos

videos = 50

try:
    native_paths = ["/data/RBC_Phantom_60xOlympus/Donor_1/Native5_focused"]
    native_videos, native_labels = get_videos(native_paths, label=1, num_videos=videos)
    print("")
    time.sleep(2)
    print("get_videos function is working")
    print("")

except Exception as e:
    print("")
    print(f"Error getting videos from paths: {e}")
    print("")

