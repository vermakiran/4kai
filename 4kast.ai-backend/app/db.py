# app/db.py
from hdbcli import dbapi
from fastapi import Depends
from .config import settings
 
def get_hana_connection():
    conn = dbapi.connect(
        address=settings.HANA_HOST,
        port=settings.HANA_PORT,
        user=settings.HANA_USER,
        password=settings.HANA_PASS
    )
 
    conn.cursor().execute("SET SCHEMA DBADMIN")

    try:
        yield conn
    finally:
        conn.close()