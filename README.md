# latent-space-investigation (v 1.0)

 ![social](https://img.shields.io/github/followers/VMrGamer?style=social)![twitter](https://img.shields.io/twitter/follow/VedantPat?style=social)![languages](https://img.shields.io/github/languages/count/VMrGamer/latent-space-investigation)

 This is a repository for my Research Methodology Course Project on, Investigation of Latent Space in Variational AutoEncoders


## Table of Contents

1.  [Manifest](#manifest)
2.  [Visuals](#visuals)
3.  [Installation](#installation)
4.  [Usage](#usage)
5.  [Support](#support)
6.  [Road Map](#road-map)
7.  [How to contribute](#how-to-contribute)
8.  [Acknowledgements](#acknowledgements)
9.  [License](#license)
10. [Project Status](#project-status)


## Manifest

- The list of the top-level files in the project.

```
- LICENSE -------> License(MIT-License) to the Project.
- README.md ------> This markdown file you are reading.
- img ----------------> images folder for README files.
- mr_citer ---------------------> The Citation Manager.
- data ------> The required data or links to access it.
- notebooks ------------> THE CODE NOTEBOOKS and notes.
- modules ---------------> helper modules for the code.
```


## Visuals

![interpolation](https://miro.medium.com/max/480/0*cYaaF2pFLECohCaI.gif)

Image generation through latent space interpolation. Source: Bilinear interpolation on latent space for random noise vectors. Figure 20

## Installation 

- You need to have Python installed, I have used the combination of JupyterLab and Google Colab but anything is possibe.
- There are three options verified by me
- JupyterLab by Jupyter, for which you can either use mamba, conda or pip

```bash
mamba install -c conda-forge jupyterlab
```

```bash
conda install -c conda-forge jupyterlab
```

```bash
pip install jupyterlab
```
here is a little note from, [JupyterLab](https://jupyter.org/install), "If installing using pip install --user, you must add the user-level bin directory to your PATH environment variable in order to launch jupyter lab. If you are using a Unix derivative (FreeBSD, GNU / Linux, OS X), you can achieve this by using export PATH="$HOME/.local/bin:$PATH" command."

- The other option is to use, classic jupyter notebooks, for which again there is mamba, conda and pip

```bash
mamba install -c conda-forge notebook
```

```bash
conda install -c conda-forge notebook
```

```bash
pip install notebook
```

- Finally, it possible to upload the notebooks on Google Colab, which is recommended course but one will need to setup the drive or even the directory structure.

- In addition one can follow and install, [Anaconda Individual Edition](https://www.anaconda.com/products/individual) for ease of access and environment setup. Also it is open source :).

- Also, it is possibe to export the .ipynb files to .py files but it is not tested, but one can raise issues if any.

- There are some parts that may need NVIDIA/apex, it will depend on your choice. 

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
```

_Note that this is for base Windows install and Linux one is pretty simple to follow as it is officially supported_

- Then there is NVIDIA/semantic-segmentation modules, from

```bash
!git clone https://github.com/NVIDIA/semantic-segmentation.git
``` 

Which can then be imported in python as followsa
```python
import sys
sys.path.insert(0,'/content/semantic-segmentation')
```

- Also runx is used as a main to NVIDIA/semantic-segmentation

```bash
pip install runx
```

- Finally we are using three libraries that may not be in the usual Development Environment

```bash
pip install git+https://github.com/tensorflow/docs
```

```bash
pip install tensorflow-probability
```


## Usage

- To use jupyter resources you can just open the command propmpt/power shell in Windows or terminal or bash shell in Linux to use the respective resources.
- For JupyterLab

```bash
jupyter lab
```

_You can also visit the [JupyterLab Documentation](https://jupyterlab.readthedocs.io/en/stable/index.html) for more information on how to get started and read the basics._

- For Classic Jupyter Notebooks

```bash
pip install runx
```

_You can also visit the [The Jupyter Notebook](https://jupyter-notebook.readthedocs.io/en/latest/?badge=latest) Documentation for more information on how to get started and read the basics._

- Or if you have installed [Anaconda Individual Edition](https://www.anaconda.com/products/individual), then just open the Anaconda Prompt and run the respective commands for client.

- MNIST Digit dataset prediction animation.

![mnist_prediction_animation](img/mnist_gif.gif)

- MNIST Digit dataset latent space.

![mnist_latent_space](img/mnist_latent_space.png)

- cifar10 dataset prediction animation.

![cifar10_prediction_animation](img/cifar10_gif.gif)

- cifar10 dataset latent space.

![cifar10_latent_space](img/cifar10_latent_space.png)

- MNIST Fashion dataset prediction animation.

![mnist_fashion_prediction_animation](img/fashion_mnist_gif.gif)

- MNIST Fashion dataset latent space.

![mnist_fashion_latent_space](img/fashion_mnist_latent_space.png)


## Support

- Contact: [email me](v.mr.gamer@gmail.com)
- For issues, raise them on the GitHub itself or mail me on the above emaill


## Road-map

- Our current goal is to get the latent space of a Variational AutoEncoders
- We can expand the work to Autoencoders and Generative Adversarial Networks
- Ambitiously, expand the work to Transformer Networks and Manifold Learners
- Superficially, Work on expanding the project into a library of sorts


## How to contribute

- To get access to contribute directly, please [email me](v.mr.gamer@gmail.com) your GitHub account and references.
- One can also fork and generate pull requests from it, please do add description to your commits tho!.


## Acknowledgements

1. [Ekin Tiu, "Understanding Latent Space in Machine Learning"](https://towardsdatascience.com/understanding-latent-space-in-machine-learning-de5a7c687d8d)
2. [Tensorflow, "Convolutional Variational Autoencoder"](https://www.tensorflow.org/tutorials/generative/cvae)


## License

- The project is Open Source, with MIT License
- The link to the License can be found [here](https://github.com/VMrGamer/latent-space-investigation/blob/main/LICENSE)


## Project Status

- The project is currently under development, under further notice.
- Feel free to contribute, generate Pull Requests or raise Issues, it is very much appreciated.