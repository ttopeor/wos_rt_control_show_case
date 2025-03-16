from setuptools import setup, find_packages

setup(
    name="wos_rt_control_show_case",               
    version="0.0",                   
    packages=find_packages('module'),   
    package_dir={'': 'module'},        
    install_requires=[
        'pyrealsense2', 'openmim', 'torch', 'opencv-python','mmdet'
    ],
    author="Yue Feng",
    author_email="ttopeor@gmail.com",
    description="Use cases of WOS rt control API",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ttopeor/wos_rt_control_show_case.git",
)
