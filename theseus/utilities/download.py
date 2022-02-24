import gdown


def download_from_drive(id_or_url, output, md5=None, quiet=False, cache=True):
    if id_or_url.startswith('http') or id_or_url.startswith('https'):
        url = id_or_url
    else:
        url = 'https://drive.google.com/uc?id={}'.format(id_or_url)

    if not cache:
        return gdown.download(url, output, quiet=quiet)
    else:
        return gdown.cached_download(url, md5=md5, quiet=quiet)
