# Deep Learning for Visual Recognition: Project

This repository contains the project for the Aarhus University course "Deep Learning for Visual Recognition".

## Project Goal

The goal of this project is to analyze and improve a YOLOv2-based object detection model for identifying champion icons on the League of Legends minimap. We will be using the "DeepLeague" dataset and codebase as our foundation. Our work will focus on reproducing the original results and then conducting experiments to improve model performance through techniques such as data augmentation and class imbalance analysis.

## Setup Instructions

This project uses Conda for environment management to ensure reproducibility.

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd dlvr-project
    ```

2.  **Create the Conda Environment:**
    Navigate to the project's root directory and run the following command. This will create a new Conda environment named `dlvr-project` with all the necessary packages defined in `environment.yml`.
    ```bash
    conda env create -f environment.yml
    ```

3.  **Activate the Environment:**
    Before running any scripts or notebooks, you must activate the environment:
    ```bash
    conda activate dlvr-project
    ```

## Data Acquisition

The dataset is not included in this repository. It must be downloaded separately.

1.  **Download the Dataset:**
    The DeepLeague dataset can be downloaded from [this link on archive.org](https://archive.org/compress/DeepLeague100K). It is a large (approx. 30GB) compressed file. We recommend using a download manager or `wget` if you are on Linux/macOS.

2.  **Extract the Data:**
    Once downloaded, extract the contents into the `data/` folder in the project's root directory. The final structure should look something like this:
    ```
    dlvr-project/
    ├── data/
    │   ├── clusters_cleaned/
    │   │   ├── test/
    │   │   ├── train/
    │   │   └── val/
    ...
    ```

## Usage

All analysis and experimentation will be conducted in Jupyter Notebooks located in the `/notebooks` directory. To start, launch Jupyter Lab from the project's root directory:

```bash
jupyter lab
```