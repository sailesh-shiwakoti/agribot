from setuptools import setup

package_name = "agribot_manipulation"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="your_name",
    maintainer_email="you@example.com",
    description="MoveIt planning scene + harvest state machine for AgriBot",
    license="MIT",
    entry_points={
        "console_scripts": [
            "planning_scene_publisher = agribot_manipulation.planning_scene_publisher:main",
            "harvest_sequencer = agribot_manipulation.harvest_sequencer:main",
        ],
    },
)
