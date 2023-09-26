"""
Created on Thu Jun 15 05:03:53 2023

Supervised DR (Dimensionality Reduction) Methods

@authors: Alexandre Levada
          Rodrigo Mendes
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def BarChartPlot(values, labels):
    plt.bar(labels, values)

    # Adding labels to the bars
    for i, value in enumerate(values):
        plt.text(i, value, str(value), ha='center', va='bottom')

    # Adding labels to the x-axis and y-axis
    plt.xlabel('Classifiers')
    plt.ylabel('Values')

    # Displaying the graph
    plt.show()
    input()
    
def GroupedBarChartPlot(values, labels, db_name, title):
    x = np.arange(len(labels))  # the label locations
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots(figsize=(10,5))

    values_qtt = len(values)
    spacing = .01
    for index, (value, label_dr)  in enumerate(values):
        if index % 2 == 1:
            rects = ax.bar((x - width/values_qtt)-spacing, value, width, label=label_dr[0])
        else:
            rects = ax.bar((x + width/values_qtt)+spacing, value, width, label=label_dr[0])
            
        spacing += .035
        autolabel(rects, ax)
        
        
    ax.set_ylabel('Scores')
    ax.set_title(db_name + ' ' + title + ' by DR')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower right', bbox_to_anchor=(1.2, 0.8))
    fig.tight_layout()
    
    # Maximize the plot window (works on some systems)
    manager = plt.get_current_fig_manager()
    if hasattr(manager, 'window'):
        manager.window.state('zoomed')
    
    plt.show()
    
    plt.savefig(f'compare_dr_methods/results/{db_name} - {title}.png', dpi=300, format='png' )
    
    

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
        
def HeatMapChartPlot(values, clf_labels, value_name, db_name, plot):
    data_types = []
    data_values = []
    
    row, column, index = plot
    
    for idx, (value, label_dr)  in enumerate(values):
        data_types.append(label_dr[0])
        data_values.append(value)
    
    plt.subplot(row, column, index)
    
    if (index - 1) // column == row - 1 and index % column == 1:
        # Condition 1: Index corresponds to the last row and the first column
        #print("Last row and first column")
        graph = sns.heatmap(data_values, annot=False, fmt=".4f", cmap="RdYlBu", xticklabels=clf_labels, yticklabels=data_types)
    elif index % column == 1 and (index - 1) // column != row - 1:
        # Condition 2: Index corresponds to the first column and not the last row
        #print("First column and not last row")
        graph = sns.heatmap(data_values, annot=False, fmt=".4f", cmap="RdYlBu", xticklabels=False, yticklabels=data_types)
    elif (index - 1) // column == row - 1 and index % column != 1:
        # Condition 3: Index corresponds to the last row and not the first column
        #print("Last row and not first column")
        graph = sns.heatmap(data_values, annot=False, fmt=".4f", cmap="RdYlBu", xticklabels=clf_labels, yticklabels=False)
    else:
        # Condition 4: Index does not match any of the specified conditions
        #print("Other position")
        graph = sns.heatmap(data_values, annot=False, fmt=".4f", cmap="RdYlBu", xticklabels=False, yticklabels=False)
      
    if graph.get_xticklabels():
        graph.set_xticklabels(graph.get_xticklabels(), rotation=30)
    
    #plt.xlabel('Classifiers')
    #plt.ylabel('Dimensionality Reduction Methods')
    #plt.title(value_name + ' by Dimensionality Reduction Methods')

    plt.ylabel(db_name)    
    plt.title(value_name)

    plt.show()
    
    plt.savefig(f'compare_dr_methods/results/grouped_bar.png', dpi=300, format='png' )