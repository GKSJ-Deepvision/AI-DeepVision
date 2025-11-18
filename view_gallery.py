"""
Simple HTTP server to view the image gallery
"""

import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

PORT = 8000
DIRECTORY = Path(__file__).parent

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

def main():
    os.chdir(DIRECTORY)
    
    handler = MyHTTPRequestHandler
    
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        url = f"http://localhost:{PORT}/image_gallery.html"
        print(f"\n{'='*70}")
        print("🌐 WEB SERVER STARTED")
        print(f"{'='*70}")
        print(f"\n📱 View gallery at: {url}")
        print(f"\n✅ Serving files from: {DIRECTORY}")
        print(f"\n🎮 Controls:")
        print(f"   • Search: Use search box to find images")
        print(f"   • Filter: All, Train, Test, High Crowd, Low Crowd")
        print(f"   • View: Click any image for full view")
        print(f"   • Sort: By filename, crowd count, or density sum")
        print(f"\n⌨️  Press Ctrl+C to stop server\n")
        print(f"{'='*70}\n")
        
        try:
            # Open browser
            print("🌍 Opening browser...")
            webbrowser.open(url)
            
            # Serve
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Server stopped")

if __name__ == '__main__':
    main()
