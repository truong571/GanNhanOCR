# host a local server to serve the web app
import http.server
import socketserver
from config import (
    HOSTING_HOST,
    HOSTING_PORT,
)

Handler = http.server.SimpleHTTPRequestHandler

# Host a local server with directory set to DATA_PATH
class CustomHandler(Handler):
    def translate_path(self, path):
        # Override the translate_path method to serve files from DATA_PATH
        path = super().translate_path(path)
        print("Path:", path)
        relpath = path[len(self.directory):]
        return relpath

# Create the server with the custom handler
# Only local host
with socketserver.TCPServer((HOSTING_HOST, HOSTING_PORT), CustomHandler) as httpd:
    print(f"Serving at port {httpd.server_address[0]}")
    print(f"Serving at port {HOSTING_PORT}")
    httpd.serve_forever()