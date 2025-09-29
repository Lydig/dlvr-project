# Pelvic MRI Segmentation in Endometriosis Project

This is the group project repository for the *Deep Learning for Visual Recognition* course at Aarhus University.  
The project focuses on applying U-Net variants to segment pelvic organs in MRI scans of patients with endometriosis.

---

## Setup Instructions

Follow these steps to set up the project environment and download the necessary data.

### 1. Prerequisites

Ensure you have the following software installed on your system:

- **Git** – For version control. [Download here](https://git-scm.com/downloads).  
- **Miniconda/Anaconda** – For managing the Python environment. [Download Miniconda here](https://docs.conda.io/en/latest/miniconda.html).

---

### 2. Clone the Repository

Open your terminal or command prompt and clone the repository:

```bash
git clone <your-repository-url>
cd dlvr-project
```

---

### 3. Create and Activate the Conda Environment

This project uses a specific set of Python packages defined in `environment.yml`.

Create the environment:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate dlvr-project
```

---

### 4. Download and Prepare the Dataset

The dataset (~8 GB) is **not** tracked by Git (see `.gitignore`).

1. Create a `data` directory (if it doesn’t exist):

   ```bash
   mkdir data
   ```

2. Download the dataset manually from [Zenodo](https://zenodo.org/records/15750762)  
   or use PowerShell:

   ```powershell
   Invoke-WebRequest -Uri "https://zenodo.org/records/15750762/files/UT-EndoMRI.zip?download=1" -OutFile "data/UT-EndoMRI.zip"
   ```

3. Unzip `UT-EndoMRI.zip` into the `data` directory.  
   Final structure should look like:

   ```
   dlvr-project/
   └── data/
       └── UT-EndoMRI/
   ```

---

### 5. Configure VS Code for Jupyter Notebooks

To use Jupyter notebooks in VS Code with this environment, register it as a kernel:

```bash
python -m ipykernel install --user --name=dlvr-project
```

**Important:** After running this, restart VS Code.

---

## How to Run

1. Activate the environment:

   ```bash
   conda activate dlvr-project
   ```

2. Open the project in VS Code:

   ```bash
   code .
   ```

3. Open a notebook from the `notebooks/` directory (e.g., `notebooks/01_initial_data_exploration.ipynb`).  
   When prompted, select the `dlvr-project` kernel. You can now run the cells.

---

## Project Structure

```
dlvr-project/
├── data/          # Contains the dataset (ignored by Git)
├── notebooks/     # Experimental/exploratory code (chronological naming: 01_..., 02_...)
├── src/           # Reusable, polished Python scripts and functions
├── material/      # Research papers, lecture notes, references
├── environment.yml # Conda environment specification
└── README.md      # This file
```

---