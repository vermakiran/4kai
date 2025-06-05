from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from app import normalize_dates
from app import clean_demand_column
from app import perform_eda
import mysql.connector
import os
from urllib.parse import unquote
import time
import csv
from io import StringIO
# Database connection
dataBase = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="4kast"
)

cursorObject = dataBase.cursor()

items = []

UPLOAD_DIR = "uploaded_files"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

class SimpleAPIHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers()

    def do_GET(self):
        if self.path == "/forecastdata":
            self._set_headers()
            self.wfile.write(json.dumps(items).encode())
        elif self.path == "/files":
            files = os.listdir(UPLOAD_DIR)
            print(f"Returning files: {files}")
            self._set_headers()
            self.wfile.write(json.dumps({"files": files}).encode())
        elif self.path.startswith("/download/"):
            filename = unquote(self.path[len("/download/"):])
            file_path = os.path.join(UPLOAD_DIR, filename)
            print(f"Requested file: {file_path}")
            if os.path.exists(file_path):
                if filename.endswith(".csv"):
                    content_type = "text/csv"
                elif filename.endswith(".xlsx") or filename.endswith(".xls"):
                    content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                else:
                    content_type = "application/octet-stream"

                self.send_response(200)
                self.send_header("Content-type", content_type)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                print(f"Served file: {file_path}")
            else:
                print(f"File not found: {file_path}")
                self._set_headers(404)
                self.wfile.write(json.dumps({"error": "File not found"}).encode())
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode())

    def do_POST(self):
        if self.path == "/forecastinput":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            item = json.loads(post_data)
            main.app(input_data=item)
            items.append(item)
            self._set_headers(201)
            self.wfile.write(json.dumps(item).encode())

        elif self.path == "/deletefile":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            item = json.loads(post_data)
            filename = item.get("filename")

            if not filename:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Filename required"}).encode())
                return

            file_path = os.path.join(UPLOAD_DIR, filename)

            if os.path.exists(file_path):
                os.remove(file_path)
                self._set_headers(200)
                self.wfile.write(json.dumps({"message": "File deleted"}).encode())
            else:
                self._set_headers(404)
                self.wfile.write(json.dumps({"error": "File not found"}).encode())

        elif self.path == "/uploadfile":
            content_length = int(self.headers["Content-Length"])
            boundary = self.headers["Content-Type"].split("boundary=")[1].encode()
            raw_data = self.rfile.read(content_length)

            parts = raw_data.split(b"--" + boundary)
            filename = None
            file_content = None
            table = None

            for part in parts:
                if b"Content-Disposition" in part:
                    if b'filename="' in part:
                        filename_start = part.find(b'filename="') + len(b'filename="')
                        filename_end = part.find(b'"', filename_start)
                        original_filename = part[filename_start:filename_end].decode()
                        base, ext = os.path.splitext(original_filename)
                        filename = f"{base}_{int(time.time())}{ext}"

                        content_start = part.find(b"\r\n\r\n") + 4
                        content_end = part.find(b"--" + boundary) - 4
                        file_content = part[content_start:content_end]
                    elif b'name="table"' in part:
                        table_start = part.find(b"\r\n\r\n") + 4
                        table_end = part.find(b"--" + boundary) - 4
                        table = part[table_start:table_end].decode()

            if filename and file_content:
                file_path = os.path.join(UPLOAD_DIR, filename)
                with open(file_path, "wb") as f:
                    f.write(file_content)
                print(f"Saved file: {file_path}, Table: {table}")
                self._set_headers(201)
                self.wfile.write(json.dumps({
                    "message": f"File '{original_filename}' uploaded as '{filename}'",
                    "filename": filename
                }).encode())
            else:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "No file found in request"}).encode())

        elif self.path == "/upload-cleaned-data":
            content_length = int(self.headers["Content-Length"])
            print(f"Content-Length: {content_length}")
            boundary = self.headers["Content-Type"].split("boundary=")[1].encode() if "boundary=" in self.headers["Content-Type"] else None
            raw_data = self.rfile.read(content_length)

            if boundary:
                parts = raw_data.split(b"--" + boundary)
                filename = None
                file_content = None

                for part in parts:
                    if b"Content-Disposition" in part and b'filename="' in part:
                        filename_start = part.find(b'filename="') + len(b'filename="')
                        filename_end = part.find(b'"', filename_start)
                        filename = part[filename_start:filename_end].decode()
                        content_start = part.find(b"\r\n\r\n") + 4
                        content_end = part.find(b"--" + boundary) - 4
                        file_content = part[content_start:content_end]

                        decoded_content = file_content.decode('utf-8')
                        csv_reader = csv.DictReader(StringIO(decoded_content))
                        json_data = [row for row in csv_reader if row]

                        # Convert to pandas DataFrame
                        import pandas as pd
                        df = pd.DataFrame(json_data)

                        # Apply cleaning functions from app.py
                        df = normalize_dates(df, "Date")
                        if df is None:
                            raise ValueError("Date normalization failed.")

                        df = clean_demand_column(df)
                        if df is None:
                            raise ValueError("Demand cleaning failed.")
                        
                        # df = perform_eda(df)
                        # if df is None:
                        #     raise ValueError("EDA failed.")

                        # Convert back to JSON if needed
                        json_string = df.to_json(orient='records', date_format='iso')
                        print(json_string)

                if filename and file_content:
                    file_path = os.path.join(UPLOAD_DIR, filename)
                    with open(file_path, "wb") as f:
                        f.write(file_content)
                    print(f"Saved cleaned file: {file_path}")
                    self._set_headers(200)
                    self.wfile.write(json.dumps({
                        "status": "success",
                        "message": f"Cleaned file '{filename}' uploaded",
                        "filename": filename
                    }).encode())
                else:
                    self._set_headers(400)
                    self.wfile.write(json.dumps({"error": "No file found in request"}).encode())
            else:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Invalid request format, expected multipart/form-data"}).encode())


        elif self.path == "/login":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            item = json.loads(post_data)
            username = item.get("username")
            password = item.get("password")

            query = "SELECT password FROM users WHERE username = %s"
            cursorObject.execute(query, (username,))
            result = cursorObject.fetchall()
            if result and len(result) > 0:
                stored_password = result[0][0]
                if stored_password == password:
                    self._set_headers(200)
                    self.wfile.write(json.dumps({"success": True, "message": "Login successful"}).encode())
                else:
                    self._set_headers(401)
                    self.wfile.write(json.dumps({"success": False, "error": "Invalid username or password"}).encode())
            else:
                self._set_headers(401)
                self.wfile.write(json.dumps({"success": False, "error": "Invalid username or password"}).encode())

        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode())

def run(server_class=HTTPServer, handler_class=SimpleAPIHandler, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server on port {port}...")
    httpd.serve_forever()

if __name__ == "__main__":
    run()