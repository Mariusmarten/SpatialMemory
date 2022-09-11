# Spatial Memory

Implementation and testing repository of several feedforward and recurrent methods for learning representations for simultaneous localisation and mapping.  
Contents:
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Usage](#usage)

## Dataset
The datasets were created using the [Gym-Miniworld](https://github.com/Farama-Foundation/MiniWorld) a minimalistic 3D interior environment simulator. The create datasets contain first-person observations, a depth map, a bird's eye view perspective, the agent coordinates, orientation and the actions performed at each location.

<p align="center">
<img src="img/environment-dataset.png"  width=50% height=50%>
</p>
  
## Models

The repository contains the following models:

![models](img/models.png "Models")

<table>
<tr>
<th>Model</th>
<th>Task</th>
<th>Type</th>
<th>Parameter</th>
<th>Description</th>
</tr>
<tr>
<td>A</td>
<td>Coordinate prediction</td>
<td>Feedforward</td>
<td>119.610</td>
<td>text</td>
</tr>
<tr>
<td>B</td>
<td>Action prediction</td>
<td>Feedforward</td>
<td>133.788</td>
<td>text</td>
</tr>
<tr>
<td>C</td>
<td>Action prediction</td>
<td>Feedforward</td>
<td>61.496</td>
<td>text</td>
</tr>
<tr>
<td>D</td>
<td>Action prediction</td>
<td>Feedforward</td>
<td>61.496</td>
<td>text</td>
</tr>
<tr>
<td>E</td>
<td>Coordinate prediction</td>
<td>Recurrent</td>
<td>366.402</td>
<td>text</td>
</tr>
<tr>
<td>F</td>
<td>Action prediction</td>
<td>Recurrent</td>
<td>365.402</td>
<td>text</td>
</tr>
<tr>
<td>G</td>
<td>Coordinate prediction</td>
<td>Recurrent</td>
<td>1.031.724</td>
<td>text</td>
</tr>
<tr>
<td>H</td>
<td>Action prediction</td>
<td>Recurrent</td>
<td>1.031.724</td>
<td>text</td>
</tr>
</table>

## Results

We find that models that integrate the largest amount of previous information perform best. Local single-state information is insufficient to distinguish identical-looking states and to uniquely determine the global position. Our results imply that model-based navigation agents profit from integrating trajectory information and that agents should thus be endowed with mechanisms to do so.

## Usage

The results reported in the report are reproducible using the provided jupyter notebook. They run with little to no dependencies. First, the dataset generation notebook needs to be ran. The obtained datasets can then be used to train the models in notebooks A-H. 
