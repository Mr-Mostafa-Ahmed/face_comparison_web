import os

port = int(os.environ.get('PORT', 10000))
bind = f"0.0.0.0:{port}"
workers = 1
threads = 1
timeout = 120
worker_class = 'sync'
max_requests = 10
max_requests_jitter = 2
preload_app = True
worker_tmp_dir = '/dev/shm'  # Use RAM for temporary files