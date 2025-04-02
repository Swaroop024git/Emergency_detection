import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'interior_monitoring'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        # Include model files
        (os.path.join('share', package_name, 'data'), 
         glob('data/*.pth')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='swaroop',
    maintainer_email='samparkranti009@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'emotion_detector = interior_monitoring.emotion_detector:main',
        ],
    },
)
