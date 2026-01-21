# Joint Reconstruction in Photoacoustic Computed Tomography

## Project Overview
Implementation of joint reconstruction (JR) method for estimating initial pressure and speed-of-sound distributions in photoacoustic tomography with object constraints.

## Features
- Joint reconstruction of IP and SOS distributions
- Implementation of support, bound, and TV constraints
- ADMM optimization algorithm
- Numerical breast phantom simulations
- Four comprehensive numerical studies

## Installation
```bash
git clone https://github.com/Serkalem-negusse1/pact-joint-reconstruction.git
cd pact-joint-reconstruction
conda env create -f environment.yml
conda activate pact-jr
pip install -e .