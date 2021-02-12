from setuptools import setup
import os


def download_tipsi_builder_module():
    import urllib.request

    remote_location = "https://gitlab.science.ru.nl/tcm/tipsi/-/raw/master/tipsi/builder.py"
    with urllib.request.urlopen(remote_location) as r:
        contents = r.read()
    pwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(pwd, "tipsi", "builder.py"), "wb") as out:
        out.write(contents)


download_tipsi_builder_module()
setup(
    name="tipsi",
    version="master",
    description="TiPSi builder module",
    packages=["tipsi"],
    install_requires=["numpy", "scipy", "matplotlib", "h5py"],
    zip_safe=True,
)
