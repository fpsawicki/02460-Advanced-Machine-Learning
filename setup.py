from setuptools import setup

setup(
        name="Lime",
        version="1.0.0",
        author="Katharina Granberg, Filip Sawicki, Javier García Ciudad, Thomas Petersen",
        packages=["lime"],
        package_dir={"lime": "lime"},
        url="https://github.com/fpsawicki/02460-Advanced-Machine-Learning",
        license="MIT",
        python_requires='>=3.5',
        install_requires=[
            "numpy",
            "matplotlib",
            "skimage",
            "sklearn",
            "nltk",
            "scipy",
            "torch",
            "torchvision",
            "torchtext",
            "pytorch_lightning"
        ]
)
