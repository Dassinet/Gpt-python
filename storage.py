import boto3
import requests
import os
from typing import List, Optional, Tuple, Union
from botocore.exceptions import ClientError
import hashlib
from urllib.parse import urlparse
import shutil
from io import BytesIO
import aiohttp
import asyncio
import aiofiles

class CloudflareR2Storage:
    def __init__(self):
        self.use_local_fallback = False
        self.r2 = None
        
        try:
            self.account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
            self.access_key = os.getenv("CLOUDFLARE_ACCESS_KEY_ID")
            self.secret_key = os.getenv("CLOUDFLARE_SECRET_ACCESS_KEY")
            self.bucket_name = os.getenv("CLOUDFLARE_BUCKET_NAME", "rag-documents")
            
            if not all([self.account_id, self.access_key, self.secret_key]):
                print("Missing Cloudflare R2 credentials. R2 services will not be available. Local fallback for KB docs only.")
                self.use_local_fallback = True
            else:
                self.r2 = boto3.resource(
                    's3',
                    endpoint_url=f'https://{self.account_id}.r2.cloudflarestorage.com',
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key
                )
                self._ensure_bucket_exists()
                
        except Exception as e:
            print(f"Error initializing R2 storage or ensuring bucket: {e}")
            print("R2 services will not be available. Falling back to local storage for KB documents only.")
            self.use_local_fallback = True
            self.r2 = None
            
        os.makedirs("local_storage/kb", exist_ok=True)

    def _ensure_bucket_exists(self) -> None:
        if not self.r2:
            raise ConnectionError("R2 client not initialized.")
        try:
            self.r2.meta.client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                print(f"Bucket '{self.bucket_name}' not found. Creating bucket.")
                self.r2.create_bucket(Bucket=self.bucket_name)
                print(f"Bucket '{self.bucket_name}' created successfully.")
            else:
                raise

    def _get_extension_from_content_type(self, content_type: str) -> str:
        content_type = content_type.lower()
        if 'pdf' in content_type: return '.pdf'
        if 'html' in content_type: return '.html'
        if 'plain' in content_type: return '.txt'
        if 'json' in content_type: return '.json'
        if 'docx' in content_type: return '.docx'
        if 'jpeg' in content_type or 'jpg' in content_type: return '.jpg'
        if 'png' in content_type: return '.png'
        return '.data'

    async def download_file_from_url(self, url: str, target_dir: str, target_filename: Optional[str] = None) -> Tuple[bool, str]:
        """
        Downloads a file from any URL to a specified local directory.
        Returns (True, local_file_path) on success, or (False, error_message) on failure.
        """
        try:
            if target_filename:
                final_filename = os.path.basename(target_filename)
            else:
                parsed_url = urlparse(url)
                basename = os.path.basename(parsed_url.path)
                if basename:
                    final_filename = basename
                else:
                    # Fallback for URLs without a clear filename
                    url_hash = hashlib.md5(url.encode()).hexdigest()
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.head(url, timeout=10) as response:
                                content_type = response.headers.get('content-type', '')
                                ext = self._get_extension_from_content_type(content_type)
                        final_filename = f"{url_hash}{ext}"
                    except Exception as head_e:
                        print(f"Could not HEAD {url} to determine file type: {head_e}. Using default.")
                        final_filename = f"{url_hash}.data"
            
            local_path = os.path.join(target_dir, final_filename)
            os.makedirs(target_dir, exist_ok=True)

            print(f"Downloading from URL '{url}' to local path '{local_path}'")
            
            timeout = aiohttp.ClientTimeout(total=120)  # Increased timeout
            headers = {'User-Agent': 'Mozilla/5.0'} # More common user agent

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        error_msg = f"Download failed with status {response.status}: {response.reason}"
                        print(f"{error_msg} for URL: {url}")
                        return False, error_msg
                    
                    async with aiofiles.open(local_path, 'wb') as f:
                        await f.write(await response.read())
                    
                    print(f"File downloaded successfully to {local_path}")
                    return True, local_path

        except Exception as e:
            error_msg = f"General error downloading URL {url}: {e}"
            print(error_msg)
            return False, error_msg


    def _upload_local_kb(self, file_data, filename: str) -> Tuple[bool, str]:
        """Upload a knowledge base file to local storage"""
        try:
            local_path = f"local_storage/kb/{filename}"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            if isinstance(file_data, bytes):
                with open(local_path, 'wb') as f:
                    f.write(file_data)
            else:
                file_data.seek(0)
                with open(local_path, 'wb') as f:
                    shutil.copyfileobj(file_data, f)
            
            file_url = f"file://{os.path.abspath(local_path)}"
            print(f"KB file '{filename}' saved locally to '{local_path}'.")
            return True, file_url
        except Exception as e:
            print(f"Error saving KB file '{filename}' locally: {e}")
            return False, str(e)

    def upload_file(self, file_data, filename: str, is_user_doc: bool = False, 
                    schedule_deletion_hours: int = 72) -> Tuple[bool, str]:
        """
        Uploads a file. If R2 is unavailable, KB files fall back to local storage.
        """
        folder = "user_docs" if is_user_doc else "kb"
        key = f"{folder}/{filename}"

        if self.use_local_fallback or not self.r2:
            if is_user_doc:
                return False, "R2 not available; user documents require R2."
            return self._upload_local_kb(file_data, filename)
        
        try:
            print(f"Uploading {'user document' if is_user_doc else 'KB file'} '{filename}' to R2 key '{key}'...")
            file_obj_to_upload = BytesIO(file_data) if isinstance(file_data, bytes) else file_data
            if hasattr(file_obj_to_upload, 'seek'):
                file_obj_to_upload.seek(0)

            self.r2.meta.client.upload_fileobj(file_obj_to_upload, self.bucket_name, key)
            file_url = f"https://{self.bucket_name}.{self.account_id}.r2.cloudflarestorage.com/{key}"
            
            self.schedule_deletion(key, schedule_deletion_hours)
            
            print(f"File uploaded to R2: {file_url} (deletion in {schedule_deletion_hours} hrs)")
            return True, file_url
        except Exception as e:
            print(f"Error uploading '{filename}' to R2: {e}. Falling back to local for KB file.")
            if not is_user_doc:
                return self._upload_local_kb(file_data, filename)
            return False, f"R2 upload failed for user document: {e}"
            
    def download_file(self, key: str, local_download_path: str) -> bool:
        """
        Download a file from R2 to a local path.
        """
        is_user_doc_key = key.startswith("user_docs/")
        is_kb_doc_key = key.startswith("kb/")

        if not is_user_doc_key and not is_kb_doc_key:
            print(f"Invalid key format: '{key}'. Must start with 'user_docs/' or 'kb/'.")
            return False

        if self.use_local_fallback or not self.r2:
            if is_user_doc_key:
                print(f"R2 unavailable. Cannot download user doc '{key}'.")
                return False
            if is_kb_doc_key:
                local_source_path = f"local_storage/{key}"
                print(f"R2 unavailable. Trying local fallback for KB file: '{local_source_path}'.")
                if os.path.exists(local_source_path):
                    try:
                        shutil.copy2(local_source_path, local_download_path)
                        print(f"KB file downloaded from local fallback to '{local_download_path}'.")
                        return True
                    except Exception as e:
                        print(f"Error copying local KB file: {e}")
                        return False
                else:
                    print(f"KB file not found in local fallback storage.")
                    return False
            return False

        try:
            print(f"Attempting to download '{key}' from R2 to '{local_download_path}'...")
            os.makedirs(os.path.dirname(local_download_path), exist_ok=True)
            self.r2.meta.client.download_file(self.bucket_name, key, local_download_path)
            print(f"File '{key}' downloaded successfully from R2.")
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if '404' in str(error_code) or 'NoSuchKey' in str(error_code):
                print(f"File '{key}' not found in R2.")
                if is_kb_doc_key:
                    local_source_path = f"local_storage/{key}"
                    print(f"Checking local fallback for KB file at '{local_source_path}'...")
                    if os.path.exists(local_source_path):
                        try:
                            shutil.copy2(local_source_path, local_download_path)
                            print(f"KB file downloaded from local fallback to '{local_download_path}'.")
                            return True
                        except Exception as e_copy:
                            print(f"Error copying local fallback KB file: {e_copy}")
                            return False
                    else:
                        print(f"KB file also not found in local fallback storage.")
                        return False
                else:
                    print(f"User document '{key}' not found in R2; no fallback.")
                    return False
            else:
                print(f"R2 ClientError when downloading '{key}': {e}")
                return False
        except Exception as e:
            print(f"General error downloading file '{key}': {e}")
            return False
    
    def list_files(self, prefix: str = "") -> List[str]:
        """List files with the given prefix."""
        if self.use_local_fallback or not self.r2:
            print(f"R2 unavailable. Listing local files with prefix '{prefix}'.")
            # This part can be improved, but is not critical for the fix
            return []
        
        try:
            print(f"Listing files from R2 bucket '{self.bucket_name}' with prefix '{prefix}'.")
            response = self.r2.meta.client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            return [obj['Key'] for obj in response.get('Contents', [])]
        except Exception as e:
            print(f"Error listing files from R2: {e}")
            return []

    def schedule_deletion(self, key: str, hours: int = 72) -> bool:
        """Schedules a file for deletion by setting object metadata."""
        if self.use_local_fallback or not self.r2:
            print(f"R2 unavailable. Cannot schedule deletion for '{key}'.")
            return False
        
        try:
            import datetime
            
            # This is the compatible way to set expiration metadata
            expiration_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=hours)
            
            # We copy the object to itself, replacing the metadata.
            self.r2.meta.client.copy_object(
                Bucket=self.bucket_name,
                CopySource={'Bucket': self.bucket_name, 'Key': key},
                Key=key,
                Metadata={
                    'delete-after': expiration_time.isoformat()
                },
                MetadataDirective='REPLACE'
            )
            
            print(f"File '{key}' marked for deletion after {expiration_time.isoformat()}.")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'NotImplemented':
                print(f"Warning: The storage provider does not support metadata updates via copy_object. Cannot schedule deletion for '{key}'.")
                return False # Gracefully fail
            print(f"Error scheduling deletion for '{key}': {e}")
            return False
        except Exception as e:
            print(f"General error scheduling deletion for '{key}': {e}")
            return False

    def check_and_delete_expired_files(self) -> int:
        """Checks file metadata and deletes expired ones."""
        if self.use_local_fallback or not self.r2:
            return 0
        
        import datetime
        deleted_count = 0
        now = datetime.datetime.now(datetime.timezone.utc)
        
        try:
            paginator = self.r2.meta.client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket_name):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    try:
                        # Check the object's metadata, not its tags
                        metadata = self.r2.meta.client.head_object(Bucket=self.bucket_name, Key=key).get('Metadata', {})
                        if 'delete-after' in metadata:
                            expiration_time = datetime.datetime.fromisoformat(metadata['delete-after'])
                            if now > expiration_time:
                                self.r2.meta.client.delete_object(Bucket=self.bucket_name, Key=key)
                                print(f"Deleted expired file '{key}'")
                                deleted_count += 1
                    except Exception as e_check:
                        print(f"Could not check/delete object '{key}': {e_check}")
            return deleted_count
        except Exception as e:
            print(f"Error checking for expired files: {e}")
            return 0

    def cleanup_expired_files(self):
        """Run periodic cleanup of expired files."""
        if self.use_local_fallback or not self.r2:
            return
        
        try:
            deleted_count = self.check_and_delete_expired_files()
            if deleted_count > 0:
                print(f"Cleanup completed: deleted {deleted_count} expired files.")
        except Exception as e:
            print(f"Error during cleanup: {e}")