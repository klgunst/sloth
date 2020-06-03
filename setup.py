from setuptools import setup
import versioneer

requirements = [
    # package requirements go here
    'numpy>=1.18',
    'h5py>=2.10',
    'networkx>=2.4'
]

setup(
    name='sloth',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="T3NS python package",
    license="MIT",
    author="sloth",
    author_email='klaasgunst@hotmail.com',
    url='https://github.com/klgunst/sloth',
    packages=['sloth'],
    
    install_requires=requirements,
    keywords='sloth',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
