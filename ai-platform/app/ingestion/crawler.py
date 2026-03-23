import boto3
import requests
from bs4 import BeautifulSoup
import os


class Crawler:
    """
    Crawl documents from:
      - S3 buckets
      - Public websites
    """

    def __init__(self, s3_bucket=None, web_urls=None):
        self.s3_bucket = s3_bucket
        self.web_urls = web_urls or []

    def crawl_s3(self):
        if not self.s3_bucket:
            return []

        s3 = boto3.client("s3")
        response = s3.list_objects_v2(Bucket=self.s3_bucket)
        files = []
        for obj in response.get("Contents", []):
            key = obj["Key"]
            file_obj = s3.get_object(Bucket=self.s3_bucket, Key=key)
            content = file_obj["Body"].read()
            files.append((key, content))
        return files

    def crawl_web(self):
        docs = []
        for url in self.web_urls:
            res = requests.get(url)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, "html.parser")
                text = soup.get_text(separator="\n")
                docs.append((url, text))
        return docs

    def crawl(self):
        s3_docs = self.crawl_s3()
        web_docs = self.crawl_web()
        return s3_docs + web_docs
