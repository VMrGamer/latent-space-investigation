# latent-space-investigation (v 1.0)

 ![social](https://img.shields.io/github/followers/VMrGamer?style=social)![twitter](https://img.shields.io/twitter/follow/VedantPat?style=social)![languages](https://img.shields.io/github/languages/count/VMrGamer/latent-space-investigation)

 This is a repository for my Research Methodology Course Project on, Investigation of Latent Space in GAN


## Table of Contents

1. [Manifest](#manifest)
2. [Installation](#installation)
3. [Support](#support)
4. [Road Map](#road-map)
5. [How to contribute](#how-to-contribute)
6. [Acknowledgements](#acknowledgements)
7. [License](#license)
8. [Project Status](#project-status)


## Manifest

- The list of the top-level files in the project.

```
- LICENSE -------> License(MIT-License) to the Project.
- README.md ------> This markdown file you are reading.
- img ----------------> images folder for README files.
- mr_citer ---------------------> The Citation Manager.
- data ------> The required data or links to access it.
- notebooks ---> THE CODE, notebooks in IPython format.
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




## Support

- Contact: [email me](v.mr.gamer@gmail.com)
- For issues, raise them on the GitHub itself or mail me on the above emaill


## Road-map

- Our current goal is to get the latent space of a Generative Adversarial Network
- We can expand the work to Autoencoders and Variational Autoencoders
- Ambitiously, expand the work to Transformer Networks and Manifold Learners
- Superficially, Work on expanding the project into a library of sorts

## How to contribute

- To get access to contribute directly, please [email me](v.mr.gamer@gmail.com) your GitHub account and references.
- One can also fork and generate pull requests from it, please do add description to your commits tho!.


## Acknowledgements

1. [Jason Brownlee, "How to Explore the GAN Latent Space When Generating Faces"](https://machinelearningmastery.com/how-to-interpolate-and-perform-vector-arithmetic-with-faces-using-a-generative-adversarial-network/)
2. [Ekin Tiu, "Understanding Latent Space in Machine Learning"](https://towardsdatascience.com/understanding-latent-space-in-machine-learning-de5a7c687d8d)


## License

- The project is Open Source, with MIT License
- The link to the License can be found [here](https://github.com/VMrGamer/latent-space-investigation/blob/main/LICENSE)


## Project Status

- The project is currently under development, and I'm working on making everything done until now on GANs more organized
- Feel free to contribute, generate Pull Requests or raise Issues, it is very much appreciated.