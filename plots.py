# -*- coding: utf-8 -*-
"""
************************************************************************
Copyright 2020 Institute of Theoretical and Applied Informatics, 
Polish Academy of Sciences (ITAI PAS) https://www.iitis.pl
author: K. Książek, P.Głomb, M. Romaszewski

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
************************************************************************

Code for experiments in the paper by  
K. Książek, M. Romaszewski, P. Głomb, B. Grabowski, M. Cholewa
`Blood Stains Classification with Hyperspectral
Imaging and Deep Neural Networks'

Files for plots for hyperblood classification/detection.
WARNING! WORKS ONLY FOR THE HYPERBLOOD DATASET!
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from data_loader import blood_loader
from data_paths import PATH_DATA

# ----------------------------------------------------------------------------

def adjust_plot(classification_type, prediction):
    """
    Labels and colors for two types of classification
    """
    if classification_type == 1:
        # for multi-class classification
        # assumption: 6 classes + background!
        color_dict = {0: [0.0, 0.0, 0.0, 1.0],
                      1: [1.0, 0.0, 0.0, 1.0],
                      2: [0.0, 0.83, 1.0, 1.0],
                      3: [0.36, 0.0, 1.0, 1.0],
                      4: [1.00, 0.62, 0.0, 1.0],
                      5: [0.05, 0.99, 0.63, 1.0],
                      6: [0.75, 0.75, 0.75, 1.0]}

        list_of_colors = [[i/6, color_dict[i]] for i in range(0, 7)]

        cmap = LinearSegmentedColormap.from_list('mycmap', list_of_colors)
        labels = {0: "background",
                  1: "blood",
                  2: "ketchup",
                  3: "artificial blood",
                  4: "poster paint",
                  5: "tomato concentrate",
                  6: "acrylic paint"}
        # colors: black, red, light blue, purple, orange, glaucous, gray

    elif classification_type == 0:
        # for one-class classification
        # assumption: 0 is ignored label, 1 is blood, 2 is background!
        color_dict = {0: [1.0, 1.0, 1.0, 1.0],
                      1: [1.0, 0.0, 0.0, 1.0],
                      2: [0.0, 0.0, 0.0, 1.0]}

        list_of_colors = [[0, color_dict[0]], [0.5, color_dict[1]], [1, color_dict[2]]]
        cmap = LinearSegmentedColormap.from_list('mycmap', list_of_colors)
        labels = {0: "ignored",
                  1: "blood",
                  2: "background"}

    return color_dict, list_of_colors, cmap, labels

# ----------------------------------------------------------------------------

def create_plots(classification_type, prediction, gt, plot_names):
    """
    Creating plots dedicated for multi-class or one-class
    classification.

    Arguments:
    classification_type: multi-class or one-class
    prediction: model prediction for pixels
    gt: ground truth with real labels
    plot_names: parameters of path and plot names
    """
    color_dict, list_of_colors, cmap, labels = adjust_plot(classification_type, prediction)
    legend_parameter = 0.75 if classification_type == 1 else 0.55

    # Plot general image
    if classification_type == 1:
        vmin = 0
        vmax = 6

    elif classification_type == 0:
        vmin = 0
        vmax = 2

    image_show = plt.imshow(prediction, cmap=cmap, vmin=vmin, vmax=vmax)
    print(prediction)
    # Create folder if don't exist
    os.makedirs("./" + plot_names['path'] + "/Binary_files/work/", exist_ok=True)
    os.makedirs("./" + plot_names['path'] + "/Images/work/", exist_ok=True)

    # save prediction as numpy array and images
    np.savez("./" + plot_names['path'] + "/Binary_files/" + plot_names['checkpoint'] + "_" +
             plot_names['dataset'] + "_prediction.npz", prediction=prediction)
    patches = [mpatches.Patch(color=color_dict[i], label=labels[i]) for i in color_dict]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, legend_parameter), loc=2, borderaxespad=0.)
    plt.savefig("./" + plot_names['path'] + "/Images/" + plot_names['checkpoint'] + "_" +
                plot_names['dataset'] + "_prediction_legend.png", bbox_inches='tight')

    # Plot image with mask (only in multiclass case)
    if classification_type == 1:
        mask = np.zeros(gt.shape, dtype='bool')
        for l in plot_names['ignored']:
            mask[gt == l] = True

        prediction[mask] = 0
        image_mask_show = plt.imshow(prediction, cmap=cmap, vmin=0, vmax=6)
        print(prediction)
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 0.75), loc=2, borderaxespad=0.)
        plt.savefig("./" + plot_names['path'] + "/Binary_files/" + plot_names['checkpoint'] + "_" +
                    plot_names['dataset'] + "_prediction_mask_legend.png", bbox_inches='tight')
        plt.savefig("./" + plot_names['path'] + "/Images/" + plot_names['checkpoint'] + "_" +
                    plot_names['dataset'] + "_prediction_mask_legend.pdf", bbox_inches='tight')

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # Ground truth plotting of comparison scene and frame scene
    # from the 1st day
    names = ['E_1', 'F_1']
    datasets = ['scene_comparison_day_1',
                'frame_day_1']

    for name, dataset in zip(names, datasets):
        _, gt, _, _, _ = blood_loader(PATH_DATA, name)
        plot_names = {'dataset': dataset,
                      'path': 'test_images',
                      'checkpoint': 'test',
                      'ignored': [0]}
        create_plots(1, gt, gt, plot_names)
