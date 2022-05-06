import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sweet-perceptron",
    version="1.0.1",
    author="SweetBubaleXXX",
    author_email="1pcpcpc1pc@gmail.com",
    license='MIT',
    description="A simple neural network library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SweetBubaleXXX/sweet-perceptron",
    project_urls={
        "Bug Tracker": "https://github.com/SweetBubaleXXX/sweet-perceptron/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=["numpy >= 1.15.0"],
    python_requires=">=3.8",
)