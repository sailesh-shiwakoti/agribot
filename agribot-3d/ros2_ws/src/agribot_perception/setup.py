from setuptools import setup

package_name = "agribot_perception"

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
    description="Red-apple detection + 3D localization for AgriBot",
    license="MIT",
    entry_points={
        "console_scripts": [
            "red_apple_detector = agribot_perception.red_apple_detector:main",
            "yolo_onnx_detector = agribot_perception.yolo_onnx_detector:main",
            "fruit_localizer = agribot_perception.fruit_localizer:main",
            "dataset_recorder = agribot_perception.dataset_recorder:main",
        ],
    },
)
