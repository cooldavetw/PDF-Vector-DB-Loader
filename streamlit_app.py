import hashlib
import io
import json
import time
import uuid
from typing import List

import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sqlalchemy import JSON, bindparam, create_engine, text
from sqlalchemy.engine import Engine
from openai import OpenAI


# ---------------------------------------------------------------------
# DB CONFIG
# ---------------------------------------------------------------------
PG_HOST = "pgvector-db"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "sEgMa6"
PG_DATABASE = "postgres"

PAGE_CONTENT_SCHEMA = "page_content"
RECORD_MANAGER_SCHEMA = "record_manager"

EMBEDDING_MODEL = "qwen3"   # OpenAI embedding model
EMBEDDING_DIM = 4096                         # Dimension for text-embedding-3-small
OPENAI_DEFAULT_BASE_URL = "http://192.168.11.20:4000/v1"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def sanitize_table_name(name: str) -> str:
    """
    Allow letters, digits, underscore, spaces, and CJK characters.
    Disallow quotes/semicolons to avoid SQL injection; return stripped name.
    """
    import re

    if not name:
        raise ValueError("Table name cannot be empty")

    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Table name cannot be empty")
    if '"' in cleaned or ";" in cleaned:
        raise ValueError('Table name cannot contain quotes or semicolons')

    allowed_pattern = r'^[A-Za-z0-9_\s\u4e00-\u9fff]+$'
    if not re.match(allowed_pattern, cleaned):
        raise ValueError(
            "Table name may contain letters, digits, underscore, spaces, and Chinese characters only"
        )
    return cleaned


def quote_table_name(name: str) -> str:
    """Return a safely double-quoted table identifier."""
    return f'"{sanitize_table_name(name)}"'


@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    driver_candidates = ["psycopg2", "psycopg"]
    engine = None
    last_error = None

    for driver in driver_candidates:
        url = f"postgresql+{driver}://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
        try:
            engine = create_engine(url, future=True)
            break
        except ImportError as exc:
            last_error = exc
            continue

    if engine is None:
        raise ImportError(
            "No PostgreSQL driver available. Install `psycopg2-binary` (preferred) or `psycopg`."
        ) from last_error

    # Initialize schemas and pgvector extension
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS page_content"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS record_manager"))
        # pgvector extension (requires superuser or proper permission)
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))

    return engine


def list_page_content_tables(engine: Engine) -> List[str]:
    sql = text(
        """
        SELECT tablename
        FROM pg_catalog.pg_tables
        WHERE schemaname = :schema
        ORDER BY tablename
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {"schema": PAGE_CONTENT_SCHEMA}).fetchall()
    return [r[0] for r in rows]


def create_tables_if_not_exist(engine: Engine, table_name: str):
    """
    Create (if not exist):

    page_content.<table_name> (
        id uuid PK DEFAULT uuid_generate_v4(),
        "pageContent" text,
        metadata jsonb,
        embedding vector
    )

    record_manager.<table_name> (
        uuid uuid PK DEFAULT gen_random_uuid(),
        "key" text NOT NULL,
        "namespace" text NOT NULL,
        updated_at float8 NOT NULL,
        group_id text NULL,
        UNIQUE("key", "namespace")
    )
    """
    table_name = sanitize_table_name(table_name)
    quoted_table = quote_table_name(table_name)

    create_page_content_sql = text(
        f"""
        CREATE TABLE IF NOT EXISTS {PAGE_CONTENT_SCHEMA}.{quoted_table} (
            id uuid NOT NULL DEFAULT uuid_generate_v4(),
            "pageContent" text NULL,
            metadata jsonb NULL,
            embedding vector NULL,
            PRIMARY KEY (id)
        )
        """
    )

    create_record_manager_sql = text(
        f"""
        CREATE TABLE IF NOT EXISTS {RECORD_MANAGER_SCHEMA}.{quoted_table} (
            uuid uuid NOT NULL DEFAULT gen_random_uuid(),
            "key" text NOT NULL,
            "namespace" text NOT NULL,
            updated_at float8 NOT NULL,
            group_id text NULL,
            UNIQUE ("key", "namespace"),
            PRIMARY KEY (uuid)
        )
        """
    )

    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {PAGE_CONTENT_SCHEMA}"))
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {RECORD_MANAGER_SCHEMA}"))
        conn.execute(create_page_content_sql)
        conn.execute(create_record_manager_sql)


def embed_texts(api_key: str, model: str, base_url: str, texts: List[str]) -> List[List[float]]:
    """
    Use OpenAI embedding API to embed a list of texts.
    """
    client = OpenAI(api_key=api_key, base_url=base_url or OPENAI_DEFAULT_BASE_URL)
    resp = client.embeddings.create(model=model, input=texts)
    # Ensure we preserve order
    return [d.embedding for d in resp.data]


def vector_to_pg_literal(vec: List[float]) -> str:
    """
    Convert a Python list[float] to a pgvector literal: '[1,2,3,...]'.
    Postgres will cast text -> vector automatically.
    """
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def extract_pdf_chunks(uploaded_file) -> List[dict]:
    """
    Chunk PDF page-by-page.
    Each chunk: {filename, page_num, content, metadata}
    """
    def _clean_text(val: str) -> str:
        # Postgres text cannot contain NUL; strip any accidental NUL bytes
        return val.replace("\x00", "")

    def _serialize_pdf_info(reader: PdfReader, filename: str) -> dict:
        info_raw = reader.metadata or {}
        info = {}
        for key, value in info_raw.items():
            name = key[1:] if isinstance(key, str) and key.startswith("/") else str(
                key
            )
            try:
                info[name] = _clean_text(str(value))
            except Exception:
                info[name] = None

        total_pages = len(reader.pages)
        version = getattr(reader, "pdf_header_version", None) or getattr(
            reader, "pdf_version", None
        )
        version_str = _clean_text(str(version)) if version is not None else ""

        return {
            "info": info,
            "version": version_str,
            "metadata": None,
            "totalPages": total_pages,
            "filename": filename,
        }

    chunks = []
    pdf_bytes = uploaded_file.read()
    reader = PdfReader(io.BytesIO(pdf_bytes))
    filename = _clean_text(uploaded_file.name)
    pdf_meta = _serialize_pdf_info(reader, filename)

    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = _clean_text(text).strip()
        if not text:
            continue
        chunks.append(
            {
                "filename": filename,
                "page_num": idx,
                "content": text,
                "metadata": {
                    "loc": {"pageNumber": idx},
                    "pdf": pdf_meta,
                    "source": "blob",
                    "blobType": "",
                },
            }
        )
    return chunks


# ---------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------
def main():
    st.title("PDF文件向量資料庫載入器")

    st.sidebar.header("Settings")
    api_key = st.sidebar.text_input(
        "OpenAI API Key (for embeddings)",
        type="password",
        help="Required to generate embeddings when upserting.",
        value="abcd"
    )
    api_base = st.sidebar.text_input(
        "OpenAI Base URL",
        value=OPENAI_DEFAULT_BASE_URL,
        help="Override for OpenAI-compatible endpoints.",
    )
    embedding_model = st.sidebar.text_input(
        "Embedding model",
        value=EMBEDDING_MODEL,
        help="Any OpenAI-compatible embedding model name.",
    )

    engine = get_engine()

    # 1) Choose / create target table
    st.subheader("1. 選擇向量資料表")
    existing_tables = list_page_content_tables(engine)
    table_options = ["<Create new table>"] + existing_tables
    table_choice = st.selectbox("Target table", table_options)

    new_table_name = ""
    if table_choice == "<Create new table>":
        new_table_name = st.text_input("New table name")

    # 2) Upload PDF
    st.subheader("2. 上傳PDF文件")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file is not None:
        if (
            "last_uploaded_name" not in st.session_state
            or st.session_state.last_uploaded_name != uploaded_file.name
        ):
            chunks = extract_pdf_chunks(uploaded_file)
            st.session_state.last_uploaded_name = uploaded_file.name
            st.session_state.chunks = chunks
        else:
            chunks = st.session_state.get("chunks", [])
    else:
        chunks = []

    # 3) Preview chunks
    st.subheader("3. 預覽資料區塊 (page-by-page)")
    if chunks:
        df_preview = pd.DataFrame(
            [
                {
                    "filename": c["filename"],
                    "page_num": c["page_num"],
                    "content_preview": c["content"][:300].replace("\n", " "),
                }
                for c in chunks
            ]
        )
        st.dataframe(df_preview, use_container_width=True)
    else:
        st.info("Upload a PDF to see extracted page chunks.")

    # 4) Upsert button
    st.subheader("4. 載入向量資料庫")

    if st.button("Upsert chunks to pgvector"):
        if not chunks:
            st.error("No chunks to upsert. Please upload a PDF first.")
            return

        # Determine final table name
        if table_choice == "<Create new table>":
            if not new_table_name.strip():
                st.error("Please provide a valid new table name.")
                return
            target_table = new_table_name.strip()
        else:
            target_table = table_choice

        try:
            target_table = sanitize_table_name(target_table)
        except ValueError as e:
            st.error(f"Invalid table name: {e}")
            return

        if not api_key:
            st.error("OpenAI API key is required to create embeddings.")
            return
        if not embedding_model.strip():
            st.error("Embedding model name is required.")
            return

        with st.spinner("Creating tables (if needed) ..."):
            create_tables_if_not_exist(engine, target_table)

        # Prepare data
        contents = [c["content"] for c in chunks]
        filenames = [c["filename"] for c in chunks]
        hashes = [
            hashlib.sha256(c.encode("utf-8")).hexdigest() for c in contents
        ]
        metadata_entries = [c["metadata"] for c in chunks]
        # Deterministic IDs let us upsert the same chunk on re-upload
        deterministic_ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, h)) for h in hashes
        ]

        # Generate embeddings
        with st.spinner("Generating embeddings ..."):
            try:
                embeddings = embed_texts(api_key, embedding_model, api_base, contents)
            except Exception as e:
                st.error(f"Error generating embeddings: {e}")
                return

        # Upsert into both schemas
        with st.spinner("Upserting into database ..."):
            pc_sql = (
                text(
                f"""
                INSERT INTO {PAGE_CONTENT_SCHEMA}.{quote_table_name(target_table)}
                    (id, "pageContent", metadata, embedding)
                VALUES
                    (:id, :page_content, CAST(:metadata AS jsonb), CAST(:embedding AS vector))
                ON CONFLICT (id) DO UPDATE
                SET
                    "pageContent" = EXCLUDED."pageContent",
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """
                )
                .bindparams(
                    bindparam("id"),
                    bindparam("page_content"),
                    bindparam("metadata", type_=JSON),
                    bindparam("embedding"),
                )
            )

            rm_sql = text(
                f"""
                INSERT INTO {RECORD_MANAGER_SCHEMA}.{quote_table_name(target_table)}
                    ("key", "namespace", updated_at, group_id)
                VALUES
                    (:key, :namespace, :updated_at, :group_id)
                ON CONFLICT ("key", "namespace") DO UPDATE
                SET
                    updated_at = EXCLUDED.updated_at,
                    group_id = EXCLUDED.group_id
                """
            )

            with engine.begin() as conn:
                for h, fname, content, meta, emb_id, emb in zip(
                    hashes, filenames, contents, metadata_entries, deterministic_ids, embeddings
                ):
                    emb_literal = vector_to_pg_literal(emb)

                    conn.execute(
                        pc_sql,
                        {
                            "id": emb_id,
                            "page_content": content,
                            "metadata": json.dumps(meta),
                            "embedding": emb_literal,
                        },
                    )

                    conn.execute(
                        rm_sql,
                        {
                            "key": h,
                            "namespace": target_table,
                            "updated_at": time.time(),
                            "group_id": fname,
                        },
                    )

        st.success(
            f"Upsert complete into page_content.\"{target_table}\" and record_manager.\"{target_table}\"."
        )


if __name__ == "__main__":
    main()
