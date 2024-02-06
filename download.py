from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io, os, pickle

# Folder ID from Google Drive
FOLDER_ID = '1v8l9oWoxY-QybUSI9kNIZdOP9avdmE9W'

# Local directory to save the downloaded files
DOWNLOAD_DIR = '/home/***REMOVED***ccnet'

# Define the scopes
SCOPES = ['https://www.googleapis.com/auth/drive']

def authenticate_google_drive_headless():
    creds = None
    # Load the credentials from the saved token
    with open('/home/***REMOVED***token2.pickle', 'rb') as token:
        creds = pickle.load(token)
    service = build('drive', 'v3', credentials=creds)
    return service

def download_file(service, file_id, file_path):
    # Check if file already exists to avoid re-downloading
    if os.path.exists(file_path):
        print(f"File already exists: {file_path}")
        return
    
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(file_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}% for {file_path}")
    except Exception as e:
        print(f"Error downloading file {file_path}: {e}")
        # Optionally, remove the partially downloaded file if there was an error
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed partially downloaded file: {file_path}")


def list_files_in_folder(service, folder_id, local_path):
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    
    page_token = None
    while True:
        response = service.files().list(q=f"'{folder_id}' in parents and trashed=false",
                                        spaces='drive',
                                        fields='nextPageToken, files(id, name, mimeType)',
                                        pageToken=page_token).execute()
        for file in response.get('files', []):
            file_path = os.path.join(local_path, file.get('name'))
            if file.get('mimeType') == 'application/vnd.google-apps.folder':
                print(f"Entering folder: {file.get('name')}")
                list_files_in_folder(service, file.get('id'), file_path)
            else:
                print(f"Downloading file: {file.get('name')}")
                download_file(service, file.get('id'), file_path)
                
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break  # Exit loop if no more files to list

def main():
    service = authenticate_google_drive_headless()
    list_files_in_folder(service, FOLDER_ID, DOWNLOAD_DIR)

if __name__ == '__main__':
    main()
