from setuptools import setup, find_packages

setup(
       name='FightingGameAI',
       version='0.1',
       packages=find_packages(),
       install_requires=[
           'pip<=23.0.1',
           'setuptools<=66',
           'wheel<=0.38.4',
           'numpy<=1.25.2',
       ],
       entry_points={
           'console_scripts': [
           ],
       },
       license='MIT',
       description='Fighting Game AI',
       long_description=open('README.md').read(),
       long_description_content_type='text/markdown',
       author='Your Name',
       author_email='neih4207@gmail.com',
       url='https://github.com/NeiH4207/FightingGameAI',
   )
