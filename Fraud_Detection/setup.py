from setuptools import find_packages, setup
setup (
    name = 'fraudlib',
    packages = find_packages(include=['fraudlib']),
    version = '0.1.0',
    description = 'Fraud detection library',
    author = 'Shivani Mittal',
    install_requires = [ 
        'os', 
        'sys', 
        'subprocess', 
        'requests', 
        'json', 
        'numpy', 
        'pandas', 
        'matplotlib', 
        'seaborn', 
        'sklearn', 
        'imblearn', 
        'tensorflow', 
        'keras', 
    ],
)