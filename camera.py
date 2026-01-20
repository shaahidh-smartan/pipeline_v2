import cv2
import gi
import threading
import time
from queue import Queue
import numpy as np

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Global GStreamer synchronization lock to prevent race conditions
_gstreamer_global_lock = threading.Lock()

class Camera:
    """Dynamic camera configuration and management class for person re-identification system."""

    def __init__(self, camera_id, config, display_width=480, display_height=480):
        """
        Initialize a camera instance.

        Args:
            camera_id (str): Unique identifier for the camera (e.g., 'cam1', 'cam2')
            config (dict): Camera configuration containing:
                - name: Display name for the camera
                - url: RTSP URL or camera source
                - width: Camera width (optional, default 640)
                - height: Camera height (optional, default 480)
            display_width (int): Width for display frame
            display_height (int): Height for display frame
        """
        self.camera_id = camera_id
        self.config = config
        self.display_width = display_width
        self.display_height = display_height

        # GStreamer components
        self.pipeline = None
        self.appsink = None

        # Frame handling
        self.frame_buffer = Queue(maxsize=2)
        self.playing = False
        self.thread = None

        # Frame processing
        self.frame_counter = 0
        self.latest_raw_frame = None
        self.latest_processed_frame = None
        self.frame_lock = threading.Lock()

        # Detection data
        self.persistent_detections = []
        self.detection_timestamp = 0

        # Aspect ratio handling
        self.original_width = config.get('width', 480)
        self.original_height = config.get('height', 480)
        self.aspect_ratio = self.original_width / self.original_height

        # Initialize GStreamer if not already done
        Gst.init(None)

        self.first_pts_ns = None         # first buffer PTS seen (nanoseconds)
        self.first_arrival_ns = None     # first wall-clock arrival time (nanoseconds)
        self.pipeline_playing_ts_ns = None  # when the pipeline entered PLAYING (wall-clock ns)


    def create_pipeline(self):
        """Create GStreamer pipeline for this camera."""
        rtsp_url = self.config.get('url', 'rtsp://localhost:8554/stream')

        pipeline_str = f"""
            rtspsrc location={rtsp_url}
                    protocols=tcp
                    ntp-sync=true
                    ntp-time-source=clock-time
                    latency=300
                    buffer-mode=4
                    drop-on-latency=true !
            rtph264depay !
            h264parse config-interval=-1 !
            nvh264dec !
            videoconvert !
            video/x-raw,format=BGR !
            queue max-size-buffers=3 max-size-time=0 max-size-bytes=0 leaky=downstream !
            appsink name=appsink_{self.camera_id}
                    emit-signals=true
                    max-buffers=2
                    drop=true
                    sync=false
        """
        pipeline_str = pipeline_str.replace('\n', ' ').strip()

        try:
            self.pipeline = Gst.parse_launch(pipeline_str)

            # --- Shared system clock (do this AFTER pipeline exists) ---
            clock = Gst.SystemClock.obtain()
            self.pipeline.use_clock(clock)
            self.pipeline.set_start_time(Gst.CLOCK_TIME_NONE)

            # -----------------------------------------------------------

            self.appsink = self.pipeline.get_by_name(f"appsink_{self.camera_id}")
            if not self.appsink:
                raise Exception(f"Could not get appsink for {self.camera_id}")

            self.appsink.connect('new-sample', self.on_new_sample)

            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect('message', self.on_bus_message)

            return True
        except Exception as e:
            print(f"Error creating pipeline for {self.camera_id}: {e}")
            return False


    def on_new_sample(self, appsink):
        """Handle new frame from GStreamer."""
        try:
            sample = appsink.emit('pull-sample')
            if sample and self.playing:
                buffer = sample.get_buffer()
                pts_ns = int(buffer.pts) if buffer.pts != Gst.CLOCK_TIME_NONE else -1
                arrive_ns = time.time_ns()
                caps = sample.get_caps()

                success, map_info = buffer.map(Gst.MapFlags.READ)
                if success:
                    structure = caps.get_structure(0)
                    width = structure.get_int('width')[1]
                    height = structure.get_int('height')[1]

                    frame_data = np.frombuffer(map_info.data, dtype=np.uint8)
                    frame = frame_data.reshape((height, width, 3)).copy()

                    # Record the first-seen timestamps (once)
                    if self.first_pts_ns is None and pts_ns >= 0:
                        self.first_pts_ns = pts_ns
                    if self.first_arrival_ns is None:
                        self.first_arrival_ns = arrive_ns

                    # Store with thread safety
                    with self.frame_lock:
                        self.latest_raw_frame = frame
                        self.frame_counter += 1
                        # Optionally store the latest PTS/arrival for quick debugging
                        self.latest_pts_ns = pts_ns
                        self.latest_arrival_ns = arrive_ns

                    # Add to processing buffer: put a record (frame + timing)
                    if self.frame_buffer.full():
                        try:
                            self.frame_buffer.get_nowait()
                        except:
                            pass

                    self.frame_buffer.put({
                        "frame": frame,
                        "pts_ns": pts_ns,
                        "arrive_ns": arrive_ns,
                        "width": width,
                        "height": height,
                        "cam": self.camera_id,
                    })

                    buffer.unmap(map_info)

        except Exception as e:
            print(f"Error in frame capture for {self.camera_id}: {e}")

        return Gst.FlowReturn.OK


    def on_bus_message(self, bus, message):
        """Handle GStreamer pipeline messages."""
        msg_type = message.type

        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Pipeline error for {self.camera_id}: {err}")
        elif msg_type == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending_state = message.parse_state_changed()
                if new_state == Gst.State.PLAYING:
                    self.playing = True
                    self.pipeline_playing_ts_ns = time.time_ns()
                    print(f"Camera {self.camera_id} pipeline started at {self.pipeline_playing_ts_ns} ns (wall clock)")

        return True

    def start(self):
        """Start the camera pipeline."""
        if not self.pipeline:
            if not self.create_pipeline():
                return False

        try:
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                print(f"Failed to start camera {self.camera_id}")
                return False
            else:
                # Set playing to True immediately instead of waiting for message bus
                self.playing = True
                print(f"Started camera {self.camera_id}")
                return True
        except Exception as e:
            print(f"Error starting camera {self.camera_id}: {e}")
            return False

    def stop(self):
        """Stop the camera pipeline."""
        self.playing = False

        try:
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)

            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=2.0)

        except Exception as e:
            print(f"Error stopping camera {self.camera_id}: {e}")

    def get_frame(self):
        """Get the latest frame from the camera buffer."""
        if not self.frame_buffer.empty():
            try:
                return self.frame_buffer.get_nowait()
            except:
                pass
        return None

    def get_latest_processed_frame(self):
        """Get the latest processed frame (with annotations)."""
        with self.frame_lock:
            return self.latest_processed_frame.copy() if self.latest_processed_frame is not None else None

    def set_processed_frame(self, frame):
        """Set the processed frame (after adding annotations)."""
        with self.frame_lock:
            self.latest_processed_frame = frame

    def get_frame_count(self):
        """Get the current frame counter."""
        with self.frame_lock:
            return self.frame_counter

    def is_playing(self):
        """Check if the camera is currently playing."""
        return self.playing

    def get_config(self):
        """Get camera configuration."""
        return self.config

    def get_camera_id(self):
        """Get camera ID."""
        return self.camera_id

    def get_name(self):
        """Get camera display name."""
        return self.config.get('name', f'Camera {self.camera_id}')

    def get_start_markers(self):
        return {
            "camera_id": self.camera_id,
            "first_buffer_pts_ns": self.first_pts_ns,      # may be -1 if camera lacks valid PTS
            "first_arrival_wall_ns": self.first_arrival_ns,
            "pts_minus_arrival_ms": (
                None if (self.first_pts_ns is None or self.first_arrival_ns is None or self.first_pts_ns < 0)
                else (self.first_pts_ns - self.first_arrival_ns) / 1e6
            ),
            "latest_pts_ns": getattr(self, "latest_pts_ns", None),
            "latest_arrival_ns": getattr(self, "latest_arrival_ns", None),
        }

class CameraManager:
    """Manages multiple cameras dynamically."""

    def __init__(self, display_width=480, display_height=480):
        """
        Initialize camera manager.

        Args:
            display_width (int): Width for display frames
            display_height (int): Height for display frames
        """
        self.cameras = {}
        self.display_width = display_width
        self.display_height = display_height
        self.running = False

    def add_camera(self, camera_id, config):
        """
        Add a new camera to the system.

        Args:
            camera_id (str): Unique identifier for the camera
            config (dict): Camera configuration

        Returns:
            bool: True if camera added successfully
        """
        if camera_id in self.cameras:
            print(f"Camera {camera_id} already exists")
            return False

        camera = Camera(camera_id, config, self.display_width, self.display_height)
        self.cameras[camera_id] = camera
        print(f"Added camera {camera_id}: {config.get('name', camera_id)}")
        return True

    def remove_camera(self, camera_id):
        """
        Remove a camera from the system.

        Args:
            camera_id (str): Camera ID to remove

        Returns:
            bool: True if camera removed successfully
        """
        if camera_id not in self.cameras:
            print(f"Camera {camera_id} not found")
            return False

        self.cameras[camera_id].stop()
        del self.cameras[camera_id]
        print(f"Removed camera {camera_id}")
        return True

    def start_all_cameras(self):
        """Start all cameras."""
        self.running = True
        success_count = 0

        for camera_id, camera in self.cameras.items():
            if camera.start():
                success_count += 1

        print(f"Started {success_count}/{len(self.cameras)} cameras")
        return success_count

    def stop_all_cameras(self):
        """Stop all cameras."""
        self.running = False

        for camera in self.cameras.values():
            camera.stop()

        print("All cameras stopped")

    def get_camera(self, camera_id):
        """Get a specific camera instance."""
        return self.cameras.get(camera_id)

    def get_all_cameras(self):
        """Get all camera instances."""
        return self.cameras

    def get_camera_ids(self):
        """Get list of all camera IDs."""
        return list(self.cameras.keys())

    def get_camera_count(self):
        """Get the number of cameras."""
        return len(self.cameras)

    def get_playing_cameras(self):
        """Get list of cameras that are currently playing."""
        return [camera_id for camera_id, camera in self.cameras.items() if camera.is_playing()]

    def print_first_pts(self):
        for cid, cam in self.cameras.items():
            m = cam.get_start_markers()
            print(cid, "first_pts_ns=", m["first_buffer_pts_ns"])

    def sample_pts_delta(self, cam_ids=None):
        cam_ids = cam_ids or self.get_camera_ids()
        recs = {}
        for cid in cam_ids:
            r = self.cameras[cid].get_frame()
            if r is None:
                print(cid, "no frame"); return
            recs[cid] = r
        pts = {cid: r.get("pts_ns", -1) for cid, r in recs.items()}
        if any(v < 0 for v in pts.values()):
            print("Some cams lack valid PTS:", pts); return
        pivot = min(pts.values())
        print("PTS deltas (ms):", {cid: (v - pivot)/1e6 for cid, v in pts.items()})

def create_camera_configs_from_ips(camera_ips, base_name="Camera"):
    """
    Create camera configurations from a list of IP addresses.

    Args:
        camera_ips (list): List of camera IP addresses or RTSP URLs
        base_name (str): Base name for cameras

    Returns:
        list: List of camera configurations
    """
    configs = []

    for i, ip in enumerate(camera_ips):
        camera_id = f"cam_{i+1}"

        # Handle different URL formats
        if ip.startswith('rtsp://'):
            url = ip
        else:
            # Assume it's just an IP and construct RTSP URL
            # This is a common format, adjust as needed for your cameras
            url = f"rtsp://admin:admin%40123@{ip}:554/stream1"

        config = {
            'name': f'{base_name} {i+1}',
            'url': url,
            'width': 480,
            'height': 480
        }

        configs.append(config)

    return configs

# Utility functions
def create_camera_structure(config):
    """Create standard camera data structure."""
    return {
        'config': config,
        'pipeline': None,
        'appsink': None,
        'frame_buffer': Queue(maxsize=2),
        'playing': False,
        'thread': None,
        'frame_counter': 0,
        'latest_raw_frame': None,
        'latest_processed_frame': None,
        'frame_lock': threading.Lock(),
        'persistent_detections': [],
        'detection_timestamp': 0,
        'original_width': config.get('width', 1280),
        'original_height': config.get('height', 720),
        'aspect_ratio': config.get('width', 1280) / config.get('height', 720)
    }

def calculate_display_dimensions(original_width, original_height, target_width, target_height):
    """Calculate dimensions to maintain aspect ratio."""
    original_aspect = original_width / original_height
    target_aspect = target_width / target_height

    if original_aspect > target_aspect:
        new_width = target_width
        new_height = int(target_width / original_aspect)
        pad_top = (target_height - new_height) // 2
        pad_bottom = target_height - new_height - pad_top
        return new_width, new_height, 0, pad_top, 0, pad_bottom
    else:
        new_height = target_height
        new_width = int(target_height * original_aspect)
        pad_left = (target_width - new_width) // 2
        pad_right = target_width - new_width - pad_left
        return new_width, new_height, pad_left, 0, pad_right, 0

def calculate_overlap(box1, box2):
    """Calculate IoU overlap between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def is_face_inside_person(face_box, person_box, margin=10):
    """Check if face is inside person bounding box."""
    fx1, fy1, fx2, fy2 = face_box
    px1, py1, px2, py2 = person_box

    px1 -= margin
    py1 -= margin
    px2 += margin
    py2 += margin

    return fx1 >= px1 and fy1 >= py1 and fx2 <= px2 and fy2 <= py2

# def create_gstreamer_pipeline(camera_id, config, on_new_sample_callback, on_bus_message_callback):
#     """Create GStreamer pipeline for camera with thread safety."""
#     rtsp_url = config.get('url', 'rtsp://localhost:8554/stream')

#     # Thread-safe pipeline creation to prevent race conditions on 4070Ti
#     with _gstreamer_global_lock:
#         # Ensure GStreamer is initialized
#         if not Gst.is_initialized():
#             Gst.init(None)

#         pipeline_str = f"""
#             rtspsrc location={rtsp_url}
#                 latency=100
#                 buffer-mode=4
#                 drop-on-latency=true
#                 do-retransmission=false !
#             rtph264depay !
#             h264parse !
#             avdec_h264
#                 max-threads=1
#                 output-corrupt=false !
#             videoconvert !
#             video/x-raw,format=BGR !
#             queue
#                 max-size-buffers=2
#                 max-size-time=0
#                 max-size-bytes=0
#                 leaky=downstream !
#             appsink name=appsink_{camera_id}
#                 emit-signals=true
#                 max-buffers=2
#                 drop=true
#                 sync=false
#         """

#         pipeline_str = pipeline_str.replace('\n', ' ').strip()

#         try:
#             pipeline = Gst.parse_launch(pipeline_str)
#             appsink = pipeline.get_by_name(f"appsink_{camera_id}")

#             if not appsink:
#                 raise Exception(f"Could not get appsink for {camera_id}")

#             # Create thread-safe callback wrapper
#             def thread_safe_callback_wrapper(appsink, camera_id):
#                 """Thread-safe wrapper for GStreamer callbacks."""
#                 try:
#                     return on_new_sample_callback(appsink, camera_id)
#                 except Exception as e:
#                     print(f"Error in thread-safe GStreamer callback for {camera_id}: {e}")
#                     return Gst.FlowReturn.OK

#             appsink.connect('new-sample', thread_safe_callback_wrapper, camera_id)

#             bus = pipeline.get_bus()
#             bus.add_signal_watch()
#             bus.connect('message', on_bus_message_callback, camera_id)

#             print(f"[GSTREAMER] Thread-safe pipeline created for {camera_id}")
#             return pipeline, appsink

#         except Exception as e:
#             print(f"[GSTREAMER] Error creating pipeline for {camera_id}: {e}")
#             raise

# Example usage
if __name__ == "__main__":
    # Example camera IPs
    camera_ips = [
        "192.168.0.106",
        "192.168.0.113",
        "192.168.0.105",
        "192.168.0.100"
    ]

    # Create configurations
    camera_configs = create_camera_configs_from_ips(camera_ips)

def create_gstreamer_pipeline(camera_id, config, on_new_sample_callback, on_bus_message_callback):
    """Create GStreamer pipeline for camera with thread safety."""
    rtsp_url = config.get('url', 'rtsp://localhost:8554/stream')

    # Thread-safe pipeline creation to prevent race conditions on 4070Ti
    with _gstreamer_global_lock:
        # Ensure GStreamer is initialized
        if not Gst.is_initialized():
            Gst.init(None)

        pipeline_str = f"""
            rtspsrc location={rtsp_url}
                latency=100
                buffer-mode=4
                drop-on-latency=true
                do-retransmission=false !
            rtph264depay !
            h264parse !
            avdec_h264
                max-threads=1
                output-corrupt=false !
            videoconvert !
            video/x-raw,format=BGR !
            queue
                max-size-buffers=2
                max-size-time=0
                max-size-bytes=0
                leaky=downstream !
            appsink name=appsink_{camera_id}
                emit-signals=true
                max-buffers=2
                drop=true
                sync=false
        """

        pipeline_str = pipeline_str.replace('\n', ' ').strip()

        try:
            pipeline = Gst.parse_launch(pipeline_str)
            appsink = pipeline.get_by_name(f"appsink_{camera_id}")

            if not appsink:
                raise Exception(f"Could not get appsink for {camera_id}")

            # Create thread-safe callback wrapper
            def thread_safe_callback_wrapper(appsink, camera_id):
                """Thread-safe wrapper for GStreamer callbacks."""
                try:
                    return on_new_sample_callback(appsink, camera_id)
                except Exception as e:
                    print(f"Error in thread-safe GStreamer callback for {camera_id}: {e}")
                    return Gst.FlowReturn.OK

            appsink.connect('new-sample', thread_safe_callback_wrapper, camera_id)

            bus = pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect('message', on_bus_message_callback, camera_id)

            print(f"[GSTREAMER] Thread-safe pipeline created for {camera_id}")
            return pipeline, appsink

        except Exception as e:
            print(f"[GSTREAMER] Error creating pipeline for {camera_id}: {e}")
            raise


    # Initialize camera manager
    manager = CameraManager(display_width=640, display_height=480)

    # Add cameras
    for i, config in enumerate(camera_configs):
        camera_id = f"cam_{i+1}"
        manager.add_camera(camera_id, config)

    print(f"Created {manager.get_camera_count()} cameras")
    for camera_id in manager.get_camera_ids():
        camera = manager.get_camera(camera_id)
        print(f"  {camera_id}: {camera.get_name()} - {camera.get_config()['url']}")
