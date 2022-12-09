# coding: utf-8
import os
import tarfile
import requests
import shutil
from supervisely_lib.io.fs import ensure_base_path, silent_remove, get_file_name, remove_dir, get_subdirs
from supervisely_lib.api.api import Api
from supervisely_lib.task.progress import Progress


def download(github_url, dest_dir, github_token=None, version="master", log_progress=True):
    tar_path = os.path.join(dest_dir, 'repo.tar.gz')
    download_tar(github_url, tar_path, github_token, version, log_progress)

    with tarfile.open(tar_path) as archive:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(archive, dest_dir)

    subdirs = get_subdirs(dest_dir)
    if len(subdirs) != 1:
        raise RuntimeError("Repo is downloaded and extracted, but resulting directory not found")
    extracted_path = os.path.join(dest_dir, subdirs[0])

    for filename in os.listdir(extracted_path):
        shutil.move(os.path.join(extracted_path, filename), os.path.join(dest_dir, filename))
    remove_dir(extracted_path)
    silent_remove(tar_path)


def download_tar(github_url, tar_path, github_token=None, version="master", log_progress=True):
    headers = {}
    if github_token is not None:
        headers = {"Authorization": "token {}".format(github_token)}

    ensure_base_path(tar_path)

    if ".git" not in github_url:
        github_url += ".git"
    tar_url = github_url.replace(".git", "/archive/{}.tar.gz".format(version))
    r = requests.get(tar_url, headers=headers, stream=True)
    if r.status_code != requests.codes.ok:
        Api._raise_for_status(r)

    progress = Progress("Downloading (KB)", len(r.content) / 1024)
    with open(tar_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            progress.iters_done_report(len(chunk) / 1024)
