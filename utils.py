import gdown

class Downloader(object):
    def __init__(self):
        pass
    def download_file(self, file_id, file_dst):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output=file_dst)
