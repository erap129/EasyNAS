from setuptools import setup
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='easynas',  # How you named your package folder (MyLib)
    packages=['easynas'],  # Chose the same as "name"
    version='0.2.1',  # Start with a small number and increase it with every change you make
    license='MIT',  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description='A simple utility for generating CNN architectures automatically, using genetic algorithms.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Elad Rapaport',  # Type in your name
    author_email='erap129@gmail.com',  # Type in your E-Mail
    url='https://github.com/erap129/EasyNAS',  # Provide either the link to your github or to your website
    download_url='https://github.com/erap129/EasyNAS/archive/0.1.tar.gz',  # I explain this later on
    keywords=['NAS', 'GENETIC', 'CNN'],  # Keywords that define your package best
    install_requires=[  # I get to this in a second
        'tqdm',
        'torch',
        'numpy',
        'pytorch_lightning',
        'scikit-learn',
        'pyeasyga'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',  # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',  # Again, pick a license
        'Programming Language :: Python :: 3'  # Specify which python versions that you want to support
    ],
)
