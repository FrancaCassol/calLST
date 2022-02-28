from setuptools import setup, find_packages

setup(
    name="calLST",
    description="code for LST calibration test",
    license="My License",
    author="Franca",
    author_email="cassol@cppm.in2p3.fr",
    version="1.0.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': ['mycode=mycode.joke:print_joke']
    }
)
