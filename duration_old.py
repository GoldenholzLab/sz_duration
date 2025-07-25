
def compare_results_old(res_by_id, res_all, agetype='',as_subplots=False):
    # Merge the results
    merged = pd.merge(res_by_id, res_all, on='Type', suffixes=('_by_id', '_all'))
    
    ## Sort by the type
    merged = merged.sort_values('Type', ascending=True)

    # Function to create horizontal bar plot
    def create_horizontal_bar_plot(data, y_column1, y_column2, title, ylabel, as_subplots,agetype):
        if as_subplots==False:
            fig, ax = plt.subplots(figsize=(12, max(8, len(data) * 0.4)))  # Adjust height based on number of types
        else:
            ax = plt.gca()

        y = np.arange(len(data))
        height = 0.35

        ax.barh(y - height/2, data[y_column1], height, label='Grouped by ID', color='lightblue')
        ax.barh(y + height/2, data[y_column2], height, label='All Seizures', color='blue')

        ax.set_yticks(y)
        ax.set_yticklabels(data['Type'])
        ax.invert_yaxis()  # labels read top-to-bottom
        
        ax.set_xlabel('Duration (minutes)')
        #ax.set_ylabel('Seizure Type')
        ax.set_title(title)
        if as_subplots==False:

            ax.legend()

        # Add gridlines
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)

        # Add value labels
        for i, (v1, v2) in enumerate(zip(data[y_column1], data[y_column2])):
            ax.text(v1, i - height/2, f'{v1:.1f}', va='center', ha='left', fontweight='bold')
            ax.text(v2, i + height/2, f'{v2:.1f}', va='center', ha='left', fontweight='bold')

        plt.title(f'{title} {agetype}')
        plt.tight_layout()
        if as_subplots==False:
            plt.show()

    if as_subplots==True:
        if agetype=='child':
            plt.subplot(2,2,1)
        else:
            plt.subplot(2,2,2)
    # Plot comparison of medians
    create_horizontal_bar_plot(merged, 'Median_by_id', 'Median_all', 
                               'Comparison of Median Durations', 'Median Duration (minutes)',
                                as_subplots=as_subplots,agetype=agetype)
    if as_subplots==True:
        if agetype=='child':
            plt.subplot(2,2,3)
        else:
            plt.subplot(2,2,4)
    # Plot comparison of 95th percentiles
    create_horizontal_bar_plot(merged, '95th_by_id', '95th_all', 
                               'Comparison of 95th Percentile Durations', '95th Percentile Duration (minutes)', as_subplots=as_subplots,agetype=agetype)



def make_boxplot_old(seizures_df,agetype='',as_subplots=False):
    # Create a copy of the DataFrame to ensure we're not modifying the original
    seizures_df = seizures_df.copy()

    # Remove rows where 'type' is NaN
    seizures_df = seizures_df.dropna(subset=['type'])

    # Ensure 'duration' is numeric and remove any NaN values
    seizures_df.loc[:, 'duration'] = pd.to_numeric(seizures_df['duration'], errors='coerce')
    seizures_df = seizures_df.dropna(subset=['duration'])

    # Filter out durations <= 0
    seizures_df = seizures_df[seizures_df['duration'] > 0]

    # Convert durations from seconds to minutes
    seizures_df.loc[:, 'duration'] = seizures_df['duration'] / 60

    # Calculate median duration for each seizure type within each Unlinked_ID
    grouped = seizures_df.groupby(['Unlinked_ID', 'type'])
    medians = grouped['duration'].median().reset_index()

    # Calculate summary statistics
    summary = medians.groupby('type')['duration'].agg(['count', 'median']).sort_values('type', ascending=False)

    # Create the plot
    if as_subplots==False:
        plt.figure(figsize=(12, max(8, len(summary) * 0.4)))
    else:
        ax = plt.gca()

    ax = sns.boxplot(y='type', x='duration', data=medians, orient='h', showfliers=False, order=summary.index)

    plt.title(f'Distribution of Median Seizure Durations by Type {agetype}')
    #plt.ylabel('Seizure Type')
    plt.xlabel('Duration (minutes)')

    # Add count as text
    for i, (idx, row) in enumerate(summary.iterrows()):
        plt.text(plt.xlim()[1], i, f"n={int(row['count'])}", 
                 verticalalignment='center', horizontalalignment='left')

    # Add median values next to the whiskers
    for i, (idx, row) in enumerate(summary.iterrows()):
        median = medians[medians['type'] == idx]['duration'].median()
        plt.text(median, i, f"{median:.2f}", 
                 verticalalignment='center', horizontalalignment='left')

    # Add sub-tic marks and sub-grid lines
    ax.xaxis.set_minor_locator(AutoMinorLocator(6))  # 6 sub-ticks between major ticks (assuming major ticks are every 5 minutes)
    ax.grid(True, axis='x', which='both', linestyle='--', alpha=0.7)
    ax.grid(True, axis='x', which='minor', linestyle=':', alpha=0.4)

    plt.tight_layout()
    if as_subplots==False:
        plt.show()

    # Print summary statistics
    #print("Summary Statistics of Median Durations by Seizure Type:")
    #print(summary)

    return medians
