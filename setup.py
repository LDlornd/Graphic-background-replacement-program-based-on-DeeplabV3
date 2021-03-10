from setuptools import setup, find_packages

setup(
    name="BApp",
    version="1.0",
    description="01_基于 Deeplab 的图片背景替换程序",
    author="Liu Haojie",
    packages=find_packages(),
    package_data={
        "": ["models/*.pth"]
    },
    scripts=[
        "run.bat"
    ]
)