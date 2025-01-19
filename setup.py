import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_desription = f.read()

__version__ = "0.0.0"

REPO_NAME = "vads-prevalent-safety-llm"
AUTHOR_USER_NAME = "kvamsi7"
SRC_REPO = "vads-mech-interp"
AUTHOR_EMAIL = "vk84351@gmail.com"

setuptools.setup(
    name = SRC_REPO,
    version = __version__,
    author= AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Research repository for exploring mechanistic interpretability and safety evaluations in large language models (LLMs), focusing on understanding SAE features and prompt safety benchmarking.",
    long_desription = long_desription,
    long_desription_content = "text/markdown",
    url = f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker" : f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir = {"":"src"},
    packages = setuptools.find_packages(where="src")
)