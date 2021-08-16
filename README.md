# latent-space-investigation (v 1.0)

 ![social](https://img.shields.io/github/followers/VMrGamer?style=social)![twitter](https://img.shields.io/twitter/follow/VedantPat?style=social)![languages](https://img.shields.io/github/languages/count/VMrGamer/research-methods-class)

 This is a repository for my Research Methodology Course Project on, Investigation of Latent Space in GAN


## Table of Contents

1. [Manifest](#manifest)
2. [Installation](#installation)
3. [Support](#support)
4. [Road-map](road-map)
5. [License](#license)
6. [Project Status](#project-status)


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
	- here is a little note from, [JupyterLab](https://jupyter.org/install), "If installing using pip install --user, you must add the user-level bin directory to your PATH environment variable in order to launch jupyter lab. If you are using a Unix derivative (FreeBSD, GNU / Linux, OS X), you can achieve this by using export PATH="$HOME/.local/bin:$PATH" command."

- There are some parts that may need NVIDIA/apex, it will depend on your choice. 

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
```
	_Note that this is for base Windows install and Linux one is pretty simple to follow as it is officially supported_

## Support

- Contact: [email me](v.mr.gamer@gmail.com)
- For issues, raise them on the GitHub itself or mail me on the above emaill

## Road-map

- Our current goal is to get the latent space of a Generative Adversarial Network
- We can expand the work to Autoencoders and Variational Autoencoders
- Ambitiously, expand the work to Transformer Networks and Manifold Learners
- Superficially, Work on expanding the project into a library of sorts


## License

- The project is Open Source, with MIT License
- The link to the License can be found [here]()

## Project Status

- The project is currently under development, and I'm working on making everything done until now on GANs more organized
- Feel free to contribute, generate Pull Requests or raise Issues, it is very much appreciated.