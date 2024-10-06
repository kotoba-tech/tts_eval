from setuptools import setup, find_packages

with open('README.md', 'r', encoding="utf-8") as f:
    readme = f.read()

version = '0.0.0'
setup(
    name='tts_eval',
    packages=find_packages(exclude=["tests", "scripts"]),
    version=version,
    license='MIT',
    description='TTS Evaluation',
    url='https://github.com/kotoba-tech/tts_eval',
    keywords=['machine-learning'],
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Asahi Ushio',
    author_email='asahi1992ushio@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',       # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    ],
    include_package_data=True,
    test_suite='tests',
    install_requires=[
        "torch",
        "numpy",
        "datasets",
        "transformers",
        "protobuf",
        "accelerate",
        "evaluate",
        "librosa",
        "soundfile",
        "jiwer",
        "pyannote.audio"
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [],
    }
)