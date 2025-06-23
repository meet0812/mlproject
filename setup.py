from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirement(file_path:str)-> List[str]:
    '''
    This function will return the list of requirmnets
    '''
    requirment = []
    with open(file_path) as file_obj:
        requirment = file_obj.readlines()
        requirment = [req.replace ("\n","")for req in requirment]
        if HYPEN_E_DOT in requirment:
            requirment.remove(HYPEN_E_DOT)
    return requirment


setup(
    name='mlproject',
    version='0.0.1',
    author='Meet',
    author_email='100meetjariwala@gmail.com',
    packages=find_packages(),
    install_requires = get_requirement('requirment.txt')
)