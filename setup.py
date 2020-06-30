from setuptools import setup


setup(
    name="bfd_desc_sims",
    packages=['bfd_desc_sims'],
    version="0.1",
    scripts=['bin/simulate.py', 'bin/submit_simulate.py']
)
