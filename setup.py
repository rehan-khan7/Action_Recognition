from setuptools import find_packages, setup

# def read_requirements():
#     with open("requirements.txt", encoding="utf8") as f:
#         return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="src",
    version="0.1",
    packages=find_packages(),
    # install_requires=read_requirements(),
    python_requires=">=3.8",
)
