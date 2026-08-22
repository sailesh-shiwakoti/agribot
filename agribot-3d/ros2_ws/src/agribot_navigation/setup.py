import os
from glob import glob
from setuptools import setup

package_name = "agribot_navigation"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="your_name",
    maintainer_email="you@example.com",
    description="SLAM + Nav2 mobile field navigation and tree detection for AgriBot",
    license="MIT",
    entry_points={
        "console_scripts": [
            "tree_detector = agribot_navigation.tree_detector:main",
            "field_coordinator = agribot_navigation.field_coordinator:main",
        ],
    },
)
