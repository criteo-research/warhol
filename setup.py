from setuptools import setup, find_packages

setup(
  name = 'warhol',
  packages = find_packages(),
  include_package_data = True,
  version = '0.14.3',
  license='MIT',
  description = 'Warhol',
  authors = 'Jules Samaran, Ugo Tanielian, Romain Beaumont',
  url = 'https://github.com/lucidrains/dalle-pytorch',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers',
    'text-to-image'
  ],
  install_requires=[
    'axial_positional_embedding',
    'DALL-E',
    'numpy', 
    'opencv-python',
    'einops>=0.3',
    'ftfy',
    'g-mlp-pytorch',
    'pillow',
    'regex',
    'taming-transformers-rom1504',
    'tokenizers',
    'torch>=1.6',
    'torchvision',
    'transformers',
    'tqdm',
    'youtokentome',
    'WebDataset', 
    'matplotlib', 
    'ipywidgets', 
    'clip-anytorch',
    'wandb'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
