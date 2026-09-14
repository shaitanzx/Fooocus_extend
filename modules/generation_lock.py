"""A shared lock that prevents image and video pipelines using the GPU together."""

import threading


generation_lock = threading.Lock()