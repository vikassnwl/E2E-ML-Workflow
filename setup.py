from setuptools import find_packages, setup


def get_requirements(file_path):
    """
    This function will return the list of requirements.
    """
    pkg_names = open(file_path).read().split("\n")
    ignore_pkg_list= ["-e ."]
    return [pkg_name for pkg_name in pkg_names if pkg_name not in ignore_pkg_list]

setup(
    name="ML Project",
    version="0.0.1",
    author="Vikas Sanwal",
    author_email="vikassnwl@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)