from setuptools import setup

def check_dependencies():
    install_requires = []
    try:
        import numpy
    except ImportError:
        install_requires.append('numpy')
    try:
        import scipy
    except ImportError:
        install_requires.append('scipy')
    try:
        import matplotlib
    except ImportError:
        install_requires.append('matplotlib')
    try:
        import pandas
    except ImportError:
        install_requires.append('pandas')
    try:
        import seaborn
    except ImportError:
        install_requires.append('seaborn')
    return install_requires

def readme():
    with open('README.rst') as f:
        return f.read()

if __name__ == "__main__":
	install_requires = check_dependencies()
	setup(
		name='isc',
		version='1.0',
		description='a psychologically-motivated agent based model of opinion change',
		url='https://github.com/psipeter/isc',
		author='Peter Duggins',
		author_email='psipeter@gmail.com',
		packages=['isc'],
		long_description=readme(),
		install_requires=install_requires,
		include_package_data=True,
		zip_safe=False
	)