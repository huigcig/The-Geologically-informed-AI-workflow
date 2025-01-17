<p align="center" width="100%">
<img src="images/workflow.jpg"  width="80%" height="80%">
</p>

<p align="center" ><b>The geologcially-informed AI workflow for automatic fully seismic stratigraphic interpretation</b></p>        

<div>
<div align="center">
    <a href='https://github.com/huigcig/' target='_blank'>Hui Gao<sup>1</sup> </a>&emsp;
    <a href='https://github.com/xinwucwp/' target='_blank'>Xinming Wu<sup>1,†,‡</sup></a>&emsp;
     <a href='https://github.com/XuesongDing' target='_blank'>Xuesong Ding <sup>2</sup></a>&emsp;
</div>
<div>
<div align="center">
    <sup>1</sup>
    University of Science and Technology of China&emsp;
    <sup>2</sup>
    The University of Texas at Austin&emsp;
    </br>
    <!-- <sup>*</sup> Equal Contribution&emsp; -->
    <sup>†</sup> Corresponding Author&emsp;
    <sup>‡</sup> Project Lead&emsp;
</div>

----------------------------------

## 🌟 An intelligent geologically-informed and data-driven approach for fully seismic stratigraphic interpretation of sedimentary basin
### 1. Stratigraphic and geophysical forward modeling
* The geological and geophysical forward modeling workflow are modified from [**ClinoformNet**](https://github.com/huigcig/ClinoformNet) (Gao et al.,GMD, 2023). [![DOI](https://img.shields.io/badge/DOI-%7Bdoi.org%2F10.5194%2Fgmd%2016%202495%202023%7D-3480bc)](https://gmd.copernicus.org/articles/16/2495/2023/)
* In the updated workflow, **geological forward modeling** contians SFM with [PyBadlands](https://github.com/badlands-model/badlands) and adding folding (& faulting) stuctures [(Wu et al., Geophysics, 2020)](https://library.seg.org/doi/10.1190/geo2019-0375.1), while **geophysical forward modeling** contains building realistic porosity model, Biot-Gassmann theory, depth-to-time conversion, and building synthetic seismic images.

<p align="center" width="100%">
<img src="images/FM.jpg"  width="90%" height="90%">
</p>

### 2. Labeled supervison and geologically-informed unsupervision
* In the labeled supervison, we use the $L_{MSE}$ and $L_{MS-SSIM}$ to train the network with labeled synthetic training datasets.
* In the **geologically-informed unsupervision**, we implement two unsupervised losses (**$L_{Isochron}$** and **$L_{Normal}$**) based on the geologically-informed priors.
* Beforing using the **$L_{Isochron}$** and **$L_{Normal}$**, we need to track the local horizon segments and estimate the normal vectors from seismic images. [ Details in the folder: {1.Strat_skeleton 2.track_local_horizon} ]

<p align="center" width="100%">
<img src="images/geologically_informed_loss.jpg"  width="60%" height="60%">
</p>

### 3. Progressive model training
<p align="center" ><b> $L_{total} = L_{MSE} + L_{MS-SSIM} + L_{Isochron} + L_{Normal}$ </b></p>

* 








 

