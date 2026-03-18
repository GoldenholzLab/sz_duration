import pandas as pd
from io import StringIO
import csv
import gc
import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from matplotlib.ticker import AutoMinorLocator
from tqdm import tqdm
import seaborn as sns
from matplotlib.ticker import AutoLocator, MultipleLocator

_STEREOTYPED_DURATION_CACHE_VERSION = 2
_STEREOTYPED_DURATION_DATE_START = pd.Timestamp('2007-11-01')
_STEREOTYPED_DURATION_DATE_END = pd.Timestamp('2024-07-17')
_STEREOTYPED_DURATION_DATETIME_REGEX = (
    r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
    r'|\d{4}-\d{2}-\d{2} \d{2}:\d{2}'
    r'|\d{1,2}/\d{1,2}/\d{2} \d{2}:\d{2}'
    r'|\d{1,2}/\d{1,2}/\d{4} \d{2}:\d{2}'
)
_STEREOTYPED_DURATION_DEFAULT_MAX_MINUTES = 24 * 60
_STEREOTYPED_DURATION_TYPE_SUBSTITUTIONS = {
    'Secondarily Generalized': 'Focal to bilateral tonic clonic*',
    'Complex Partial': 'Focal impaired awareness*',
    'Aura Only': 'Focal preserved consciousness*',
    'Other': 'Unknown*',
    'Unknown': 'Unknown*',
    'Simple Partial': 'Focal preserved consciousness*',
}

def read_seizuretracker_csv(file_path):
    print(f"Attempting to read file: {file_path}")
    
    # Read the entire file content
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        print("File read successfully")
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None

    # Split the content into sections
    sections = content.split('\n\n')
    print(f"Number of sections found: {len(sections)}")

    # Process Profiles section
    try:
        profiles_section = next(section for section in sections if section.startswith('Profiles'))
        profiles_lines = profiles_section.split('\n')[1:-1]  # Exclude 'Profiles' and 'End Profiles' lines
        profiles_df = pd.read_csv(StringIO('\n'.join(profiles_lines)), skipinitialspace=True)
        print(f"Profiles DataFrame shape: {profiles_df.shape}")
    except Exception as e:
        print(f"Error processing Profiles section: {e}")
        return None, None

    # Process Seizures section
    try:
        seizures_section = next(section for section in sections if section.startswith('Seizures'))
        seizures_lines = seizures_section.split('\n')[1:]  # Exclude 'Seizures' line
        seizures_df = pd.read_csv(StringIO('\n'.join(seizures_lines)), skipinitialspace=True, low_memory=False)
        print(f"Seizures DataFrame shape: {seizures_df.shape}")
    except Exception as e:
        print(f"Error processing Seizures section: {e}")
        return None, None

    # Remove duplicate columns from seizures_df
    seizures_df = seizures_df.loc[:, ~seizures_df.columns.duplicated()]
    print(f"Seizures DataFrame shape after removing duplicates: {seizures_df.shape}")

    # Convert Date_Time to datetime, trying multiple formats
    def parse_date(date_string):
        for fmt in ('%Y-%m-%d %H:%M:%S', '%m/%d/%y %H:%M', '%Y-%m-%d %H:%M', '%m/%d/%Y %H:%M'):
            try:
                parsed_date = pd.to_datetime(date_string, format=fmt)
                if pd.Timestamp.min < parsed_date < pd.Timestamp.max:
                    return parsed_date
            except ValueError:
                pass
        print(f"Unable to parse date or date out of valid range: {date_string}")
        return pd.NaT

    print("Converting dates...")
    seizures_df['Date_Time'] = seizures_df['Date_Time'].apply(parse_date)
    print("Date conversion complete")

    # Filter based on date range
    print("Filtering dates...")
    date_mask = (seizures_df['Date_Time'] >= '2007-11-01') & (seizures_df['Date_Time'] <= '2024-07-17')
    seizures_df = seizures_df.loc[date_mask]

    # compute duration in seconds
    seizures_df['duration'] = seizures_df['length_hr']*60*60 + seizures_df['length_min']*60 + seizures_df['length_sec']

    print(f"Seizures DataFrame shape after date filtering: {seizures_df.shape}")

    # Print information about invalid dates
    invalid_dates = seizures_df[seizures_df['Date_Time'].isna()]
    print(f"Number of invalid dates: {len(invalid_dates)}")
    if len(invalid_dates) > 0:
        print("Sample of rows with invalid dates:")
        print(invalid_dates.head())

    # Define seizure type substitutions
    type_substitutions = {
        "Secondarily Generalized": "Focal to bilateral tonic clonic*",
        "Complex Partial": "Focal impairedconsciousness*",
        "Aura Only": "Focal preserved consciousness*",
        "Other": "Unknown*",
        "Unknown": "Unknown*",
        "Simple Partial": "Focal preserved consciousness*"
    }
    # Apply substitutions
    seizures_df['type'] = seizures_df['type'].replace(type_substitutions)

    return profiles_df, seizures_df

def age_cut(seizures_df, profiles_df):
    # Create copies of the DataFrames to avoid SettingWithCopyWarning
    seizures = seizures_df.copy()
    profiles = profiles_df.copy()
    
    # Ensure 'Date_Time' in seizures_df and 'Birth_Date' in profiles_df are datetime
    seizures['Date_Time'] = pd.to_datetime(seizures['Date_Time'], errors='coerce')
    profiles['Birth_Date'] = pd.to_datetime(profiles['Birth_Date'], errors='coerce')
    
    # Merge seizures with profiles on Unlinked_ID
    merged = pd.merge(seizures, profiles[['Unlinked_ID', 'Birth_Date']], on='Unlinked_ID', how='left')
    
    # Calculate age at time of seizure
    merged['Age_at_Seizure'] = (merged['Date_Time'] - merged['Birth_Date']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    
    # Filter out invalid ages (negative or extremely high)
    merged = merged[(merged['Age_at_Seizure'] >= 0) & (merged['Age_at_Seizure'] <= 120)]
    
    # Separate into child and adult seizures
    child_seizures = merged[merged['Age_at_Seizure'] < 18].drop(columns=['Birth_Date', 'Age_at_Seizure'])
    adult_seizures = merged[merged['Age_at_Seizure'] >= 18].drop(columns=['Birth_Date', 'Age_at_Seizure'])
    
    # Print summary statistics
    print(f"Total seizures: {len(merged)}")
    print(f"Child seizures (0-17 years): {len(child_seizures)}")
    print(f"Adult seizures (18+ years): {len(adult_seizures)}")
    print(f"Seizures excluded due to invalid age: {len(seizures) - len(merged)}")
    id_child = child_seizures['Unlinked_ID'].nunique()
    id_adult = adult_seizures['Unlinked_ID'].nunique()
    print(f"Number of unique patients: {profiles['Unlinked_ID'].nunique()} with {id_child} children and {id_adult} adults")
    overalap = set(child_seizures['Unlinked_ID'].unique()).intersection(set(adult_seizures['Unlinked_ID'].unique()))
    print(f"Overlap between children and adults: {len(overalap)}")
    
    return child_seizures, adult_seizures

def drawGraph(seizures_df, limx=10, agetype='',as_subplots=False,thetype='all',output_file='output_file.csv'):
    if thetype == 'all':
        sub_df = seizures_df.copy()
    else:
        sub_df = seizures_df[seizures_df['type'] == thetype].copy()
    
    # Filter durations greater than 0 and convert to minutes
    filtered_sub = sub_df[sub_df['duration'] > 0].copy()
    durations = pd.to_numeric(filtered_sub['duration'], errors='coerce') / 60
    durations = durations.dropna()

    if len(durations) == 0:
        n_patients = filtered_sub['Unlinked_ID'].nunique() if 'Unlinked_ID' in filtered_sub.columns else 0
        txt = f"all,{thetype},{agetype},0,{n_patients},nan,nan"
        print(txt)
        with open(output_file, 'a') as file:
            file.write(txt + '\n')
        return

    # Sort the durations and calculate the cumulative probabilities
    #print(f"Number of seizures: {len(durations)} for number of patients: {len(sub_df['Unlinked_ID'].unique())}")
    sorted_durations = np.sort(durations.to_numpy())
    cumulative_probs = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)
    # Find the index of the duration closest to 5 minutes

    #index5 = np.abs(sorted_durations - 5).argmin()
    
    # Get the cumulative probability at that index
    #cumulative_prob5 = cumulative_probs[index5]
    
    index90 = np.abs(cumulative_probs - 0.9).argmin()
    cumulative_prob90 = cumulative_probs[index90]
    prob90_time = sorted_durations[index90]

    # Create the plot
    if as_subplots==False:
        plt.figure(figsize=(4, 6))
    

    if agetype == 'child':
        thecolor = 'blue'
    else:
        thecolor = 'red'

    plt.plot(sorted_durations, cumulative_probs * 100, label=agetype, color=thecolor)
    
    plt.axhline(y=90, linestyle='--', linewidth=2, color='gray')
    plt.axvline(x=prob90_time, linestyle='--', linewidth=2, color=thecolor)
    texty = 20 if thecolor == 'red' else 10
    haChoice = 'left' if thecolor == 'red' else 'right'
    textx = prob90_time + (-1 if thecolor == 'red' else 1)
    if prob90_time > 10:
        textx += (-1 if thecolor == 'red' else 1)
    plt.text(textx, texty, f'{prob90_time:.0f}', ha=haChoice, va='center', color=thecolor)
    #plt.text(prob90_time, cumulative_prob90 * 100, f'{prob90_time}', ha='right', va='bottom')

    #print(f'{cumulative_prob90} % at {prob90_time} minutes for {agetype}')

    txt = f"all,{thetype},{agetype},{len(durations)},{len(filtered_sub['Unlinked_ID'].unique())},{prob90_time:.1f},{np.median(sorted_durations):.1f}"
    print(txt)
    # Open the file in append mode, which creates the file if it does not exist
    with open(output_file, 'a') as file:
        file.write(txt + '\n')

    #print(f"Sz n={len(durations)} pts n=:{len(sub_df['Unlinked_ID'].unique())} cdf90={prob90_time} for {agetype}")
    
    #plt.axvline(x=5, linestyle='--', linewidth=2)
    #plt.text(5, cumulative_prob5 * 100, f'{cumulative_prob5:.0%}', ha='right', va='bottom')
    #print(f'{cumulative_prob5} % at 5minutes for {agetype}')

    # Set tick marks 
    limx = np.max([limx, int(prob90_time*1.1)])


    # Set the x-axis limit to limx minutes
    plt.xlim(0, limx)
    
    # Set the y-axis limit from 0 to 100%
    plt.ylim(0, 100)
    
    # Labels and title
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Cumulative Probability (%)')
    if as_subplots==False:
        plt.title(f'CDF of Seizure Durations {agetype}')
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)


    if limx < 10:
        xstep = 1
    elif limx > 30:
        xstep = 10
    elif limx > 20:
        xstep = 5
    else:
        xstep = 2
    plt.xticks(np.arange(0, limx+1, xstep))  # X-axis ticks from 0 to 10, every 1 minute
    plt.yticks(np.arange(0, 101, 10))  # Y-axis ticks from 0% to 100%, every 10%
    
    # Set minor tick marks
    ax = plt.gca()  # Get the current Axes instance
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))  # Set minor ticks for x-axis, between the major ticks
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))  # Set minor ticks for y-axis, between the major ticks
    # Add minor grid lines
    ax.grid(True, linestyle=':', alpha=0.5, which='minor')  # Minor grid lines

    if as_subplots==False:
        # Show the plot
        plt.show()

def drawGraph_population(seizures_df, limx=10, agetype='', as_subplots=False, thetype='all',output_file='output_file.csv'):
    if thetype == 'all':
        sub_df = seizures_df.copy()
    else:
        sub_df = seizures_df[seizures_df['type'] == thetype].copy()
    
    # Filter durations greater than 0 and convert to minutes
    filtered_sub = sub_df[sub_df['duration'] > 0].copy()
    #durations = filtered_sub['duration'] / 60

    if thetype == 'all':
        # Group by Unlinked_ID and type, then calculate median duration for each group
        patient_medians = filtered_sub.groupby(['Unlinked_ID'])['duration'].median().reset_index()
    else:
        # Group by Unlinked_ID and type, then calculate median duration for each group
        patient_medians = filtered_sub.groupby(['Unlinked_ID', 'type'])['duration'].median().reset_index()

    # Convert durations to minutes
    patient_medians['duration'] = patient_medians['duration'] / 60
    
    # Filter durations greater than 0
    durations = pd.to_numeric(patient_medians[patient_medians['duration'] > 0]['duration'], errors='coerce')
    durations = durations.dropna()

    if len(durations) == 0:
        n_patients = filtered_sub['Unlinked_ID'].nunique() if 'Unlinked_ID' in filtered_sub.columns else 0
        txt = f"pickone,{thetype},{agetype},0,{n_patients},nan,nan"
        print(txt)
        with open(output_file, 'a') as file:
            file.write(txt + '\n')
        return

    # Sort the durations and calculate the cumulative probabilities
    sorted_durations = np.sort(durations.to_numpy())
    cumulative_probs = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)
    
    ## Find the index of the duration closest to 5 minutes
    #index5 = np.abs(sorted_durations - 5).argmin()
    
    ## Get the cumulative probability at that index
    #cumulative_prob5 = cumulative_probs[index5]
    index90 = np.abs(cumulative_probs - 0.9).argmin()
    cumulative_prob90 = cumulative_probs[index90]
    prob90_time = sorted_durations[index90]

    # Create the plot
    if not as_subplots:
        plt.figure(figsize=(4, 6))
    

    if agetype == 'child':
        thecolor = 'blue'
    else:
        thecolor = 'red'

    plt.plot(sorted_durations, cumulative_probs * 100, label=agetype, color=thecolor)
    
    plt.axhline(y=90, linestyle='--', linewidth=2, color='gray')
    plt.axvline(x=prob90_time, linestyle='--', linewidth=2, color=thecolor)
    
    texty = 20 if thecolor == 'red' else 10
    haChoice = 'left' if thecolor == 'red' else 'right'
    textx = prob90_time + (-1 if thecolor == 'red' else 1) 
    if prob90_time > 10:
        textx += (-1 if thecolor == 'red' else 1)
    plt.text(textx, texty, f'{prob90_time:.0f}', ha=haChoice, va='center', color=thecolor)

    #plt.text(prob90_time, cumulative_prob90 * 100, f'{prob90_time}', ha='right', va='bottom')
    #print(f"Number of seizures: {len(durations)} for number of patients: {len(sub_df['Unlinked_ID'].unique())} {prob90_time} minutes for {agetype}")
    txt = f"pickone,{thetype},{agetype},{len(durations)},{len(filtered_sub['Unlinked_ID'].unique())},{prob90_time:.1f},{np.median(sorted_durations):.1f}"
    print(txt)
    # Open the file in append mode, which creates the file if it does not exist
    with open(output_file, 'a') as file:
        file.write(txt + '\n')

#    print(f"{agetype},{len(durations)},{len(sub_df['Unlinked_ID'].unique())},{prob90_time}")

    #plt.axvline(x=5, linestyle='--', linewidth=2)
    #plt.text(5, cumulative_prob5 * 100, f'{cumulative_prob5:.0%}', ha='right', va='bottom')
    #print(f'{cumulative_prob5} % at 5minutes for {agetype}')
    
    # Set the x-axis limit to limx minutes
    plt.xlim(0, limx)
    
    # Set the y-axis limit from 0 to 100%
    plt.ylim(0, 100)
    
    # Labels and title
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Cumulative Probability (%)')
    if not as_subplots:
        plt.title(f'CDF of Median Seizure Durations {agetype}')
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Set tick marks 
    #plt.xticks(np.arange(0, limx+1, limx/10))  # X-axis ticks from 0 to limx, every limx/10 minutes
    limx = np.max([limx, int(prob90_time*1.1)])
    
    if limx < 10:
        xstep = 1
    elif limx > 30:
        xstep = 10
    elif limx > 20:
        xstep = 5
    else:
        xstep = 2
    plt.xticks(np.arange(0,limx+1,xstep))
    plt.yticks(np.arange(0, 101, 10))  # Y-axis ticks from 0% to 100%, every 10%
    
    # Set minor tick marks
    ax = plt.gca()  # Get the current Axes instance
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))  # Set minor ticks for x-axis, between the major ticks
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))  # Set minor ticks for y-axis, between the major ticks
    
    # Add minor grid lines
    ax.grid(True, linestyle=':', alpha=0.5, which='minor')  # Minor grid lines

    if not as_subplots:
        # Show the plot
        plt.show()
        
def process_unlinked_id(unlinked_id, group):
    valid_seizures = group[(group['duration'] > 0) & (group['duration'].notna())]
    
    if valid_seizures.empty:
        return unlinked_id, pd.Series(dtype=float)
    
    id_medians = valid_seizures.groupby('type')['duration'].median()
    
    if (id_medians < 1).any():
        print(f"Warning: Unlinked_ID {unlinked_id} has median duration < 1 second for types: {id_medians[id_medians < 1].index.tolist()}")
    
    return unlinked_id, id_medians

def analyze_seizure_durations(seizures_df):
    # Create a copy of the DataFrame to avoid modifying the original
    seizures_df = seizures_df.copy()
    
    # Remove rows where 'type' is NaN
    seizures_df = seizures_df.dropna(subset=['type'])

    # Ensure 'duration' is numeric and remove any NaN values
    seizures_df.loc[:, 'duration'] = pd.to_numeric(seizures_df['duration'], errors='coerce')
    seizures_df = seizures_df.dropna(subset=['duration'])

    # Get all seizure types
    all_types = seizures_df['type'].unique()

    # Group the dataframe by Unlinked_ID
    grouped = seizures_df.groupby('Unlinked_ID')

    # Process each group
    results = {}
    for unlinked_id, group in tqdm(grouped, desc="Processing Unlinked_IDs"):
        unlinked_id, id_medians = process_unlinked_id(unlinked_id, group)
        results[unlinked_id] = id_medians

    # Create DataFrame from results
    median_df = pd.DataFrame.from_dict(results, orient='index', columns=all_types)

    # Calculate statistics for each seizure type
    stats = []
    for seizure_type in all_types:
        type_data = median_df[seizure_type].dropna()
        
        if not type_data.empty:
            stats.append({
                'Type': seizure_type,
                'Min': type_data.min(),
                'Median': type_data.median(),
                'Max': type_data.max(),
                '95th': type_data.quantile(0.95),
                'Count': len(type_data)  # Number of Unlinked_IDs with this seizure type
            })

    # Create a DataFrame from the results
    result_df = pd.DataFrame(stats)

    # Sort the DataFrame by the median duration
    result_df = result_df.sort_values('Median', ascending=False)

    # Convert durations from seconds to minutes
    for col in ['Min', 'Median', 'Max', '95th']:
        result_df[col] = result_df[col] / 60

    # Round all numeric columns to 2 decimal places
    result_df = result_df.round(2)

    return result_df


def analyze_durations_all(seizures_df):
    # Create a copy of the DataFrame to avoid modifying the original
    seizures_df = seizures_df.copy()
    
    # Remove rows where 'type' is NaN
    seizures_df = seizures_df.dropna(subset=['type'])

    # Ensure 'duration' is numeric and remove any NaN values
    seizures_df.loc[:, 'duration'] = pd.to_numeric(seizures_df['duration'], errors='coerce')
    seizures_df = seizures_df.dropna(subset=['duration'])

    # Filter out durations <= 0
    seizures_df = seizures_df[seizures_df['duration'] > 0]

    # Get all seizure types
    all_types = seizures_df['type'].unique()

    # Calculate statistics for each seizure type
    stats = []
    for seizure_type in all_types:
        type_data = seizures_df[seizures_df['type'] == seizure_type]['duration']
        
        if not type_data.empty:
            stats.append({
                'Type': seizure_type,
                'Min': type_data.min(),
                'Median': type_data.median(),
                'Max': type_data.max(),
                '95th': type_data.quantile(0.95),
                'Count': len(type_data)  # Number of seizures of this type
            })

    # Create a DataFrame from the results
    result_df = pd.DataFrame(stats)

    # Sort the DataFrame by the median duration
    result_df = result_df.sort_values('Median', ascending=False)

    # Convert durations from seconds to minutes
    for col in ['Min', 'Median', 'Max', '95th']:
        result_df[col] = result_df[col] / 60

    # Round all numeric columns to 2 decimal places
    result_df = result_df.round(2)

    return result_df

def compare_results(res_by_id, res_all, ax1, ax2, agetype=''):
    # Merge the results
    merged = pd.merge(res_by_id, res_all, on='Type', suffixes=('_by_id', '_all'))
    
    # Sort by the 'Grouped by ID' median values
    merged = merged.sort_values('Median_by_id', ascending=True)

    y = np.arange(len(merged))
    height = 0.35

    # Plot median durations
    ax1.barh(y - height/2, merged['Median_by_id'], height, label='Grouped by ID', color='lightblue')
    ax1.barh(y + height/2, merged['Median_all'], height, label='All Seizures', color='blue')

    ax1.set_yticks(y)
    ax1.set_yticklabels(merged['Type'])
    ax1.invert_yaxis()  # labels read top-to-bottom
    
    ax1.set_xlabel('Median Duration (minutes)')
    #ax1.set_ylabel('Seizure Type')
    ax1.set_title(f'{agetype} Seizures: Comparison of Median Durations')
    ax1.set_xlim([0,8])
    ax1.legend()

    # Add gridlines
    ax1.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Add value labels for median
    for i, (v1, v2) in enumerate(zip(merged['Median_by_id'], merged['Median_all'])):
        ax1.text(v1, i - height/2, f'{v1:.1f}', va='center', ha='left', fontweight='bold')
        ax1.text(v2, i + height/2, f'{v2:.1f}', va='center', ha='left', fontweight='bold')

    # Plot 95th percentile durations
    ax2.barh(y - height/2, merged['95th_by_id'], height, label='Grouped by ID', color='pink')
    ax2.barh(y + height/2, merged['95th_all'], height, label='All Seizures', color='red')

    ax2.set_yticks(y)
    ax2.set_yticklabels(merged['Type'])
    ax2.invert_yaxis()  # labels read top-to-bottom
    
    ax2.set_xlabel('95th Percentile Duration (minutes)')
    #ax2.set_ylabel('Seizure Type')
    ax2.set_title(f'{agetype} Seizures: Comparison of 95th Percentile Durations')
    ax2.legend()

    # Add gridlines
    ax2.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Add value labels for 95th percentile
    for i, (v1, v2) in enumerate(zip(merged['95th_by_id'], merged['95th_all'])):
        ax2.text(v1, i - height/2, f'{v1:.1f}', va='center', ha='left', fontweight='bold')
        ax2.text(v2, i + height/2, f'{v2:.1f}', va='center', ha='left', fontweight='bold')

    ax2.set_xlim([0,300])
    



def make_boxplot(child_seizures, adult_seizures):
    # Prepare data for children
    child_df = prepare_data_for_boxplot(child_seizures, 'Child')
    
    # Prepare data for adults
    adult_df = prepare_data_for_boxplot(adult_seizures, 'Adult')
    
    # Combine the dataframes
    combined_df = pd.concat([child_df, adult_df])
    
    # Calculate summary statistics
    summary = combined_df.groupby(['type', 'age_group'])['duration'].agg(['count', 'median']).reset_index()
    summary = summary.sort_values(by=['age_group', 'type'], ascending=[False, True])
        
    # Create the boxplot
    ax = sns.boxplot(y='type', x='duration', hue='age_group', data=combined_df, 
                     orient='h', showfliers=False, palette=['blue', 'red'])
    
    plt.title('One per patient median durations by type: Children vs Adults')
    plt.xlabel('Duration (minutes)')
    
    # Add count and median as text
    for i, (_, row) in enumerate(summary.iterrows()):
        y_pos = i - 0.2 if row['age_group'] == 'Child' else i + 0.2
    #    plt.text(plt.xlim()[1], y_pos, 
    #             f"{row['age_group'][0]}:n={int(row['count'])}, m={row['median']:.2f}", 
    #             verticalalignment='center', horizontalalignment='left', fontsize=8)
    
    # Add sub-tic marks and sub-grid lines
    ax.xaxis.set_minor_locator(AutoMinorLocator(6))
    ax.grid(True, axis='x', which='both', linestyle='--', alpha=0.7)
    ax.grid(True, axis='x', which='minor', linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    

def prepare_data_for_boxplot(seizures_df, age_group):
    df = seizures_df.copy()
    df = df.dropna(subset=['type'])
    df.loc[:, 'duration'] = pd.to_numeric(df['duration'], errors='coerce')
    df = df.dropna(subset=['duration'])
    df = df[df['duration'] > 0]
    df.loc[:, 'duration'] = df['duration'] / 60
    medians = df.groupby(['Unlinked_ID', 'type'])['duration'].median().reset_index()
    medians['age_group'] = age_group
    return medians


def make_table_1(seizures_df, profiles_df, csv_file):
    # Get unique Unlinked_ID values from seizures dataframe
    unique_ids = seizures_df['Unlinked_ID'].unique()
    
    # Filter profiles dataframe to include only patients in the seizures dataframe
    relevant_profiles = profiles_df[profiles_df['Unlinked_ID'].isin(unique_ids)].copy()
    
    # Calculate total number of profiles
    total_profiles = len(relevant_profiles)
    
    # Calculate age
    seizures_df['Date_Time'] = pd.to_datetime(seizures_df['Date_Time'], errors='coerce')
    median_seizure_dates = seizures_df.groupby('Unlinked_ID')['Date_Time'].median()
    
    # Handle invalid birth dates
    relevant_profiles['Birth_Date'] = pd.to_datetime(relevant_profiles['Birth_Date'], errors='coerce')
    
    # Calculate age, handling NaT values and excluding negative ages
    relevant_profiles['Median_Seizure_Date'] = relevant_profiles['Unlinked_ID'].map(median_seizure_dates)
    relevant_profiles['Age'] = (relevant_profiles['Median_Seizure_Date'] - relevant_profiles['Birth_Date']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    
    # Exclude negative ages
    relevant_profiles = relevant_profiles[relevant_profiles['Age'] >= 0]
    
    # Recalculate total number of profiles after age exclusion
    total_profiles_after_age_filter = len(relevant_profiles)
    
    # Summarize age
    age_summary = relevant_profiles['Age'].agg(['count', 'mean', 'std', 'min', 'max'])
    
    # Summarize gender
    gender_summary = relevant_profiles['Gender'].value_counts()
    gender_percentages = (gender_summary / total_profiles_after_age_filter * 100).round(1)
    
    # List of medical conditions to count
    conditions = [
        'Congenital_Condition', 'Brain_Tumors', 'Brain_Trauma', 'Brain_Trauma_Hematomas',
        'Infectious_Diseases', 'Alzheimers', 'Stroke', 'Heart_Attack', 'Brain_Surgery',
        'Metabolic_Disorder', 'Genetic_Abnormalities', 'Electrolyte_Disturbances',
        'Brain_Injury_During_Fetal_Development', 'Lack_Of_Oxygen_During_Birth',
        'Maternal_Drug_Or_Alcohol_Abuse', 'Brain_Malformations', 'Lead_Exposure',
        'Carbon_Monoxide_Exposure', 'Alcohol_Or_Drug_Abuse', 'High_Fever'
    ]
    
    # Count medical conditions (only for columns that exist)
    existing_conditions = [col for col in conditions if col in relevant_profiles.columns]
    condition_counts = relevant_profiles[existing_conditions].notna().sum()
    condition_percentages = (condition_counts / total_profiles_after_age_filter * 100).round(1)
    
    # Prepare data for summary DataFrame
    data = {
        'Characteristic': ['N (total profiles before age filter)', 'N (total profiles after age filter)', 'Age count', 'Age mean (years)', 'Age std (years)', 'Age min (years)', 'Age max (years)'] +
                          [f'Gender: {gender}' for gender in gender_summary.index] +
                          [f'Condition: {condition}' for condition in conditions],
        'Value': [
            f"{total_profiles:.0f}",
            f"{total_profiles_after_age_filter:.0f}",
            f"{age_summary['count']:.0f}",
            f"{age_summary['mean']:.2f}",
            f"{age_summary['std']:.2f}",
            f"{age_summary['min']:.2f}",
            f"{age_summary['max']:.2f}"
        ] +
        [f"{count:.0f} ({percentage:.1f}%)" for count, percentage in zip(gender_summary.values, gender_percentages.values)] +
        [f"{condition_counts.get(condition, 0):.0f} ({condition_percentages.get(condition, 0):.1f}%)" for condition in conditions]
    }
    
    # Create summary DataFrame
    summary = pd.DataFrame(data)
    summary.to_csv(csv_file, index=False, quoting=csv.QUOTE_NONNUMERIC)

    return summary

def bigCDF(seizures_df, max_duration=None,agetype='',as_subplots=False):
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df = seizures_df.copy()
    
    # Ensure 'duration' is in minutes
    df['duration_minutes'] = df['duration'] / 60
    
    if max_duration:
        # Filter out durations greater than max_duration
        df = df[df['duration_minutes'] <= max_duration]
    
    # Filter out durations <1 second
    df = df[df['duration_minutes'] > 1/60]
    
    # Replace 'nan' with 'Unknown*' in the 'type' column
    df['type'] = df['type'].fillna('Unknown*')
    
    # Calculate 90th percentile duration for each seizure type
    percentile_90 = df.groupby('type')['duration_minutes'].quantile(0.9).sort_values()
    
    # Get sorted unique seizure types
    seizure_types = percentile_90.index.tolist()
    
    if as_subplots==False:
        # Set up the plot
        plt.figure(figsize=(5, 12))

    # Color map (red to blue)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(seizure_types)))
    
    # Markers
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
    
    # Plot CDF for each seizure type
    for i, seizure_type in enumerate(seizure_types):
        type_data = df[df['type'] == seizure_type]['duration_minutes']
        
        # Sort the durations and calculate the cumulative probabilities
        sorted_durations = np.sort(type_data)
        cumulative_probs = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)
        
        # Plot the CDF
        plt.plot(sorted_durations, cumulative_probs * 100, 
                 color=colors[i], marker=markers[i % len(markers)], 
                 linestyle='-', markersize=4, markevery=0.1, 
                 label=f"{seizure_type} (90%ile: {percentile_90[seizure_type]:.2f})")
    
    # Customize the plot
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Cumulative Probability (%)')
    plt.title(f'CDF Durations by Type, {agetype}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, 30)
    plt.ylim(0, 100)
    
    # Add legend
    plt.legend(title='Seizure Type', loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.legend(title='Seizure Type', loc='lower right')
    
    # Adjust layout to prevent cutoff
    plt.tight_layout()
    
    if as_subplots==False:
        # Show the plot
        plt.show()


def makeBoxPlotsV2(seizures_child, seizures_adult):
    # Prepare data for children
    child_df = prepare_data_for_plot(seizures_child, 'Child')
    
    # Prepare data for adults
    adult_df = prepare_data_for_plot(seizures_adult, 'Adult')
    
    # Combine the dataframes
    combined_df = pd.concat([child_df, adult_df])
    
    # Calculate summary statistics
    summary = combined_df.groupby(['type', 'age_group'])['duration'].agg(['count', 'median']).reset_index()
    summary = summary.sort_values(by=['type', 'age_group'], ascending=[True, False])
    
    # Set up the plot
    plt.figure(figsize=(12, 15))
    
    # Create the boxplot
    ax = sns.boxplot(y='type', x='duration', hue='age_group', data=combined_df, 
                     orient='h', width=0.8, fliersize=2,
                     palette={'Child': 'blue', 'Adult': 'red'})
    
    plt.title('Seizure Durations by Type: Children vs Adults', fontsize=16)
    plt.ylabel('Seizure Type', fontsize=12)
    plt.xlabel('Duration (minutes)', fontsize=12)
    
    # Set x-axis to log scale
    ax.set_xscale('log')
    
    # Add count and median as text
    for i, (_, row) in enumerate(summary.iterrows()):
        x_pos = ax.get_xlim()[1]
        y_pos = i + (0.2 if row['age_group'] == 'Child' else -0.2)
        plt.text(x_pos, y_pos, f"{row['age_group'][0]}:n={int(row['count'])}, m={row['median']:.1f}", 
                 verticalalignment='center', horizontalalignment='left', fontsize=8)
    
    # Adjust layout and display the plot
    #plt.tight_layout()
    plt.savefig('Fig4-median_simpler-boxplot.png',dpi=300)
    plt.show()

def prepare_data_for_plot(seizures_df, age_group):
    df = seizures_df.copy()
    df = df.dropna(subset=['type'])
    df.loc[:, 'duration'] = pd.to_numeric(df['duration'], errors='coerce')
    df = df.dropna(subset=['duration'])
    df = df[df['duration'] > 0]
    df.loc[:, 'duration'] = df['duration'] / 60  # Convert to minutes
    df['age_group'] = age_group
    return df

def drawSimplerBoxes(res_child, res_adult):
    print(res_child.head())
    print(res_adult.head())

    # Prepare data
    child_data = prepare_data_for_simplerboxplot(res_child, 'Child')
    adult_data = prepare_data_for_simplerboxplot(res_adult, 'Adult')
    combined_data = pd.concat([child_data, adult_data])

    # Identify the column representing the central tendency
    numeric_columns = child_data.select_dtypes(include=[np.number]).columns
    central_tendency_col = next((col for col in numeric_columns if col not in ['Count']), None)
    
    if central_tendency_col is None:
        raise ValueError("Could not find a suitable numeric column for central tendency")

    # Sort data by adult central tendency
    type_order = adult_data.groupby('Type')[central_tendency_col].median().sort_values(ascending=True).index

    # Set up the plot
    plt.figure(figsize=(12, 10))
    
    # Create the boxplot
    ax = sns.boxplot(y='Type', x=central_tendency_col, hue='Age Group', data=combined_data, 
                     orient='h', order=type_order, palette={'Child': 'lightblue', 'Adult': 'pink'})

    # Customize the plot
    plt.title(f'{central_tendency_col} Seizure Durations by Type: Children vs Adults', fontsize=16)
    plt.xlabel(f'{central_tendency_col} Duration (minutes)', fontsize=12)
    plt.ylabel('Seizure Type', fontsize=12)
    
    # Set x-axis to log scale
    ax.set_xscale('log')
    
    # Add count information
    for i, seizure_type in enumerate(type_order):
        child_count = res_child[res_child['Type'] == seizure_type]['Count'].values[0]
        adult_count = res_adult[res_adult['Type'] == seizure_type]['Count'].values[0]
        
        plt.text(ax.get_xlim()[1], i, f'C:{child_count}, A:{adult_count}', 
                 verticalalignment='center', horizontalalignment='left', fontsize=8)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

def prepare_data_for_simplerboxplot(res_df, age_group):
    # Identify columns representing statistics (excluding 'Type' and 'Count')
    stat_columns = [col for col in res_df.columns if col not in ['Type', 'Count']]

    # Melt the dataframe to create a format suitable for boxplot
    melted_df = pd.melt(res_df, id_vars=['Type', 'Count'], 
                        value_vars=stat_columns, 
                        var_name='Statistic', value_name='Value')
    
    # Add age group information
    melted_df['Age Group'] = age_group
    
    return melted_df

def make_a_pair_of_CDFs(child_seizures, adult_seizures, thetype='all',titlestr='All types',fname='Fig1-CDF-all.png',output_file='CDF_data.csv',xlimlist=(10,18)):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    drawGraph(child_seizures,agetype='child',as_subplots=True,thetype=thetype,output_file=output_file)
    #plt.subplot(1,2,2)
    drawGraph(adult_seizures,agetype='adult',as_subplots=True,thetype=thetype,output_file=output_file)
    plt.title(titlestr)
    plt.legend()
    #plt.savefig('Fig1-CDF-all.png',dpi=300)
    #plt.show()

    #  first
    #plt.figure(figsize=(3,3))
    plt.subplot(1,2,2)
    drawGraph_population(child_seizures,agetype='child',as_subplots=True,limx=xlimlist[0],thetype=thetype,output_file=output_file)
    #plt.subplot(1,2,2)
    drawGraph_population(adult_seizures,agetype='adult',as_subplots=True,limx=xlimlist[1],thetype=thetype,output_file=output_file)
    plt.ylabel('') # remove xlabel
    #plt.yticks([])  # Remove y-axis tick labels

    plt.title(f'(1 per patient)')
    #plt.legend()
    #plt.savefig('Fig1b-CDF-pop.png',dpi=300)
    plt.tight_layout()
    plt.savefig(fname,dpi=300)
    plt.show()



def _resolve_stereotyped_duration_stage_dir(stage_dir, st_csv_path):
    if stage_dir is None:
        csv_path = Path(st_csv_path).expanduser()
        if csv_path.exists():
            return csv_path.resolve().parent / '.stereotyped_duration_cache'
        return Path.cwd() / '.stereotyped_duration_cache'

    resolved_stage_dir = Path(stage_dir).expanduser()
    if not resolved_stage_dir.is_absolute():
        resolved_stage_dir = Path.cwd() / resolved_stage_dir
    return resolved_stage_dir


def _read_json_if_exists(file_path):
    if not file_path.exists():
        return None
    with open(file_path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _write_json_atomic(file_path, payload):
    temp_path = file_path.with_name(f'{file_path.name}.tmp')
    with open(temp_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temp_path.replace(file_path)


def _read_pickle_if_exists(file_path):
    if not file_path.exists():
        return None
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)


def _write_pickle_atomic(file_path, payload):
    temp_path = file_path.with_name(f'{file_path.name}.tmp')
    with open(temp_path, 'wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temp_path.replace(file_path)


def _save_npz_atomic(file_path, **arrays):
    temp_path = file_path.with_name(f'{file_path.name}.tmp')
    with open(temp_path, 'wb') as handle:
        np.savez_compressed(handle, **arrays)
    temp_path.replace(file_path)


def _file_signature(file_path):
    path = Path(file_path).expanduser()
    if not path.exists():
        return None

    stat = path.stat()
    return {
        'path': str(path.resolve()),
        'size': int(stat.st_size),
        'mtime_ns': int(stat.st_mtime_ns),
    }


def _build_stereotyped_duration_source_signature(
    source_kind,
    st_csv_path,
    profiles_pkl_path,
    seizures_pkl_path,
    chunk_rows,
    max_duration_minutes,
):
    return {
        'version': _STEREOTYPED_DURATION_CACHE_VERSION,
        'source_kind': source_kind,
        'chunk_rows': int(chunk_rows),
        'max_duration_minutes': None if max_duration_minutes is None else float(max_duration_minutes),
        'date_start': str(_STEREOTYPED_DURATION_DATE_START),
        'date_end': str(_STEREOTYPED_DURATION_DATE_END),
        'csv_source': _file_signature(st_csv_path),
        'profiles_pickle': _file_signature(profiles_pkl_path),
        'seizures_pickle': _file_signature(seizures_pkl_path),
    }


def _default_stereotyped_duration_progress(source_kind, source_signature, chunk_rows):
    return {
        'version': _STEREOTYPED_DURATION_CACHE_VERSION,
        'source_kind': source_kind,
        'source_signature': source_signature,
        'chunk_rows': int(chunk_rows),
        'profiles_ready': False,
        'processed_chunk_count': 0,
        'source_rows_processed': 0,
        'valid_rows_aggregated': 0,
        'aggregation_complete': False,
        'final_ready': False,
        'last_completed_stage': None,
    }


def _initialize_stereotyped_duration_stage_dir(
    stage_dir,
    source_kind,
    source_signature,
    chunk_rows,
    resume,
):
    progress_path = stage_dir / 'progress.json'

    if not resume and stage_dir.exists():
        shutil.rmtree(stage_dir)

    stage_dir.mkdir(parents=True, exist_ok=True)
    progress = _read_json_if_exists(progress_path)

    if progress is None or progress.get('source_signature') != source_signature:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)
        progress = _default_stereotyped_duration_progress(source_kind, source_signature, chunk_rows)
        _write_json_atomic(progress_path, progress)

    return progress


def _locate_seizuretracker_sections(file_path):
    sections = {
        'profiles_line': None,
        'profiles_header_line': None,
        'end_profiles_line': None,
        'seizures_line': None,
        'seizures_header_line': None,
    }
    waiting_for_profiles_header = False
    waiting_for_seizures_header = False

    with open(file_path, 'r', newline='') as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped_line = raw_line.strip()

            if stripped_line == 'Profiles':
                sections['profiles_line'] = line_number
                waiting_for_profiles_header = True
                continue

            if waiting_for_profiles_header and stripped_line:
                sections['profiles_header_line'] = line_number
                waiting_for_profiles_header = False
                continue

            if stripped_line == 'End Profiles':
                sections['end_profiles_line'] = line_number
                continue

            if stripped_line == 'Seizures':
                sections['seizures_line'] = line_number
                waiting_for_seizures_header = True
                continue

            if waiting_for_seizures_header and stripped_line:
                sections['seizures_header_line'] = line_number
                waiting_for_seizures_header = False
                break

    missing_sections = [name for name, value in sections.items() if value is None]
    if missing_sections:
        raise ValueError(f'Unable to locate required export sections: {missing_sections}')

    return sections


def _extract_profiles_birth_lookup_from_csv(file_path):
    birth_lookup = {}
    in_profiles_section = False
    header_row = None
    unlinked_id_index = None
    birth_date_index = None

    with open(file_path, 'r', newline='') as handle:
        for raw_line in handle:
            stripped_line = raw_line.strip()

            if not in_profiles_section:
                if stripped_line == 'Profiles':
                    in_profiles_section = True
                continue

            if stripped_line == 'End Profiles':
                break

            if not stripped_line:
                continue

            row = next(csv.reader([raw_line]))

            if header_row is None:
                header_row = [value.strip() for value in row]
                unlinked_id_index = header_row.index('Unlinked_ID')
                birth_date_index = header_row.index('Birth_Date')
                continue

            if len(row) <= max(unlinked_id_index, birth_date_index):
                continue

            unlinked_id = str(row[unlinked_id_index]).strip()
            birth_date_text = str(row[birth_date_index]).strip()
            parsed_birth_date = pd.to_datetime(birth_date_text, errors='coerce')

            if not unlinked_id or pd.isna(parsed_birth_date):
                continue

            birth_lookup[unlinked_id] = pd.Timestamp(parsed_birth_date)

    return birth_lookup


def _extract_profiles_birth_lookup_from_pickle(profiles_pkl_path):
    profiles = pd.read_pickle(profiles_pkl_path)
    required_columns = {'Unlinked_ID', 'Birth_Date'}
    if not required_columns.issubset(set(profiles.columns)):
        raise ValueError('profiles pickle is missing required columns for stereotyped-duration analysis')

    profiles = profiles[['Unlinked_ID', 'Birth_Date']].copy()
    profiles['Unlinked_ID'] = profiles['Unlinked_ID'].astype('string').str.strip()
    profiles['Birth_Date'] = pd.to_datetime(profiles['Birth_Date'], errors='coerce')
    profiles = profiles.dropna(subset=['Unlinked_ID', 'Birth_Date'])

    birth_lookup = dict(zip(profiles['Unlinked_ID'], profiles['Birth_Date']))

    del profiles
    gc.collect()
    return birth_lookup


def _parse_seizuretracker_datetime_series(date_series):
    if pd.api.types.is_datetime64_any_dtype(date_series):
        return pd.to_datetime(date_series, errors='coerce')

    cleaned_dates = date_series.astype('string').str.strip()
    candidate_mask = cleaned_dates.str.fullmatch(_STEREOTYPED_DURATION_DATETIME_REGEX, na=False)
    parsed_dates = pd.Series(pd.NaT, index=cleaned_dates.index, dtype='datetime64[ns]')

    for date_format in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%m/%d/%y %H:%M', '%m/%d/%Y %H:%M'):
        unresolved_mask = parsed_dates.isna() & candidate_mask
        if not unresolved_mask.any():
            break
        parsed_dates.loc[unresolved_mask] = pd.to_datetime(
            cleaned_dates.loc[unresolved_mask],
            format=date_format,
            errors='coerce',
        )

    return parsed_dates


def _compute_stereotyped_duration_seconds(seizure_chunk):
    if 'duration' in seizure_chunk.columns:
        return pd.to_numeric(seizure_chunk['duration'], errors='coerce')

    required_columns = {'length_hr', 'length_min', 'length_sec'}
    if not required_columns.issubset(set(seizure_chunk.columns)):
        raise ValueError("'duration' column missing and could not be computed from length_* columns")

    duration_hours = pd.to_numeric(seizure_chunk['length_hr'], errors='coerce').fillna(0.0)
    duration_minutes = pd.to_numeric(seizure_chunk['length_min'], errors='coerce').fillna(0.0)
    duration_seconds = pd.to_numeric(seizure_chunk['length_sec'], errors='coerce').fillna(0.0)
    return duration_hours * 60 * 60 + duration_minutes * 60 + duration_seconds


def _empty_stereotyped_duration_chunk():
    return pd.DataFrame(columns=['age_group', 'Unlinked_ID', 'type', 'duration'])


def _prepare_stereotyped_duration_chunk(seizure_chunk, birth_lookup, max_duration_seconds):
    if seizure_chunk.empty:
        return _empty_stereotyped_duration_chunk(), {
            'source_rows': 0,
            'retained_rows': 0,
            'invalid_datetime_rows': 0,
            'over_duration_rows': 0,
        }

    source_rows = len(seizure_chunk)
    seizure_chunk = seizure_chunk.copy()
    seizure_chunk.columns = [column.strip() for column in seizure_chunk.columns]
    seizure_chunk['Unlinked_ID'] = seizure_chunk['Unlinked_ID'].astype('string').str.strip()
    seizure_chunk = seizure_chunk[seizure_chunk['Unlinked_ID'].notna() & (seizure_chunk['Unlinked_ID'] != '')].copy()

    if seizure_chunk.empty:
        return _empty_stereotyped_duration_chunk(), {
            'source_rows': source_rows,
            'retained_rows': 0,
            'invalid_datetime_rows': 0,
            'over_duration_rows': 0,
        }

    parsed_datetimes = _parse_seizuretracker_datetime_series(seizure_chunk['Date_Time'])
    valid_datetime_mask = (
        parsed_datetimes.notna()
        & (parsed_datetimes >= _STEREOTYPED_DURATION_DATE_START)
        & (parsed_datetimes <= _STEREOTYPED_DURATION_DATE_END)
    )
    invalid_datetime_rows = int((~valid_datetime_mask).sum())

    if not valid_datetime_mask.any():
        return _empty_stereotyped_duration_chunk(), {
            'source_rows': source_rows,
            'retained_rows': 0,
            'invalid_datetime_rows': invalid_datetime_rows,
            'over_duration_rows': 0,
        }

    seizure_chunk = seizure_chunk.loc[valid_datetime_mask].copy()
    parsed_datetimes = parsed_datetimes.loc[valid_datetime_mask]

    durations = _compute_stereotyped_duration_seconds(seizure_chunk)
    valid_duration_mask = durations.notna() & (durations > 0)
    if max_duration_seconds is None:
        over_duration_rows = 0
    else:
        over_duration_rows = int((valid_duration_mask & (durations > max_duration_seconds)).sum())
        valid_duration_mask = valid_duration_mask & (durations <= max_duration_seconds)
    if not valid_duration_mask.any():
        return _empty_stereotyped_duration_chunk(), {
            'source_rows': source_rows,
            'retained_rows': 0,
            'invalid_datetime_rows': invalid_datetime_rows,
            'over_duration_rows': over_duration_rows,
        }

    seizure_chunk = seizure_chunk.loc[valid_duration_mask].copy()
    parsed_datetimes = parsed_datetimes.loc[valid_duration_mask]
    durations = durations.loc[valid_duration_mask]

    seizure_types = seizure_chunk['type'].astype('string').str.strip()
    seizure_types = seizure_types.replace(_STEREOTYPED_DURATION_TYPE_SUBSTITUTIONS)
    seizure_types = seizure_types.fillna('Unknown*').replace({'': 'Unknown*'})

    birth_dates = seizure_chunk['Unlinked_ID'].map(birth_lookup)
    birth_dates = pd.to_datetime(birth_dates, errors='coerce')
    age_years = (parsed_datetimes - birth_dates).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    valid_age_mask = birth_dates.notna() & age_years.notna() & age_years.between(0, 120)

    if not valid_age_mask.any():
        return _empty_stereotyped_duration_chunk(), {
            'source_rows': source_rows,
            'retained_rows': 0,
            'invalid_datetime_rows': invalid_datetime_rows,
            'over_duration_rows': over_duration_rows,
        }

    age_groups = np.where(age_years.loc[valid_age_mask] < 18, 'child', 'adult')
    working_chunk = pd.DataFrame(
        {
            'age_group': pd.Categorical(age_groups, categories=['child', 'adult']),
            'Unlinked_ID': seizure_chunk.loc[valid_age_mask, 'Unlinked_ID'].astype('string').to_numpy(),
            'type': pd.Categorical(seizure_types.loc[valid_age_mask].to_numpy()),
            'duration': durations.loc[valid_age_mask].astype('float64').to_numpy(),
        }
    )

    return working_chunk, {
        'source_rows': source_rows,
        'retained_rows': int(len(working_chunk)),
        'invalid_datetime_rows': invalid_datetime_rows,
        'over_duration_rows': over_duration_rows,
    }


def _merge_grouped_stereotyped_duration_stats(stats_by_key, grouped_stats):
    for row in grouped_stats.itertuples(index=False):
        age_group = str(row.age_group)
        unlinked_id = str(row.Unlinked_ID)
        seizure_type = str(row.type)
        key = (age_group, unlinked_id, seizure_type)

        new_count = int(row.count)
        new_mean = float(row.mean)
        new_variance = 0.0 if pd.isna(row.var) else float(row.var)
        new_m2 = new_variance * max(new_count - 1, 0)

        if key not in stats_by_key:
            stats_by_key[key] = (new_count, new_mean, new_m2)
            continue

        existing_count, existing_mean, existing_m2 = stats_by_key[key]
        combined_count = existing_count + new_count
        mean_delta = new_mean - existing_mean
        combined_mean = existing_mean + mean_delta * new_count / combined_count
        combined_m2 = (
            existing_m2
            + new_m2
            + mean_delta * mean_delta * existing_count * new_count / combined_count
        )
        stats_by_key[key] = (combined_count, combined_mean, combined_m2)


def _aggregate_stereotyped_duration_stats_from_csv(
    st_csv_path,
    birth_lookup,
    stage_dir,
    progress,
    chunk_rows,
    max_duration_seconds,
):
    progress_path = stage_dir / 'progress.json'
    stats_state_path = stage_dir / 'stats_state.pkl'
    stats_by_key = _read_pickle_if_exists(stats_state_path)
    if not isinstance(stats_by_key, dict):
        stats_by_key = {}

    sections = progress.get('sections')
    if sections is None:
        sections = _locate_seizuretracker_sections(st_csv_path)
        progress['sections'] = sections
        _write_json_atomic(progress_path, progress)

    seizure_chunks = pd.read_csv(
        st_csv_path,
        skiprows=int(sections['seizures_line']),
        usecols=['Unlinked_ID', 'Date_Time', 'length_hr', 'length_min', 'length_sec', 'type'],
        dtype={'Unlinked_ID': 'string', 'Date_Time': 'string', 'type': 'string'},
        chunksize=chunk_rows,
        skipinitialspace=True,
        low_memory=False,
    )

    for chunk_index, seizure_chunk in enumerate(seizure_chunks):
        if chunk_index < int(progress.get('processed_chunk_count', 0)):
            continue

        working_chunk, chunk_summary = _prepare_stereotyped_duration_chunk(
            seizure_chunk,
            birth_lookup,
            max_duration_seconds,
        )
        grouped_stats = None
        if not working_chunk.empty:
            grouped_stats = (
                working_chunk.groupby(['age_group', 'Unlinked_ID', 'type'], observed=True)['duration']
                .agg(count='count', mean='mean', var='var')
                .reset_index()
            )
            _merge_grouped_stereotyped_duration_stats(stats_by_key, grouped_stats)

        progress['processed_chunk_count'] = chunk_index + 1
        progress['source_rows_processed'] = int(progress.get('source_rows_processed', 0)) + chunk_summary['source_rows']
        progress['valid_rows_aggregated'] = int(progress.get('valid_rows_aggregated', 0)) + chunk_summary['retained_rows']
        progress['invalid_datetime_rows'] = int(progress.get('invalid_datetime_rows', 0)) + chunk_summary['invalid_datetime_rows']
        progress['over_duration_rows'] = int(progress.get('over_duration_rows', 0)) + chunk_summary['over_duration_rows']
        progress['last_completed_stage'] = 'aggregate_chunk'

        _write_pickle_atomic(stats_state_path, stats_by_key)
        _write_json_atomic(progress_path, progress)

        print(
            f"  completed chunk {chunk_index + 1}: kept {chunk_summary['retained_rows']:,} "
            f"of {chunk_summary['source_rows']:,} seizure rows"
        )

        del seizure_chunk
        del working_chunk
        if grouped_stats is not None:
            del grouped_stats
        gc.collect()

    progress['aggregation_complete'] = True
    progress['last_completed_stage'] = 'aggregate'
    _write_pickle_atomic(stats_state_path, stats_by_key)
    _write_json_atomic(progress_path, progress)
    return stats_by_key, progress


def _aggregate_stereotyped_duration_stats_from_pickle(
    seizures_pkl_path,
    birth_lookup,
    stage_dir,
    progress,
    chunk_rows,
    max_duration_seconds,
):
    progress_path = stage_dir / 'progress.json'
    stats_state_path = stage_dir / 'stats_state.pkl'
    stats_by_key = _read_pickle_if_exists(stats_state_path)
    if not isinstance(stats_by_key, dict):
        stats_by_key = {}

    seizures = pd.read_pickle(seizures_pkl_path)
    seizures.columns = [column.strip() for column in seizures.columns]

    needed_columns = ['Unlinked_ID', 'Date_Time', 'type']
    if 'duration' in seizures.columns:
        needed_columns.append('duration')
    else:
        needed_columns.extend(['length_hr', 'length_min', 'length_sec'])

    processed_chunk_count = int(progress.get('processed_chunk_count', 0))
    start_row = processed_chunk_count * chunk_rows
    total_rows = len(seizures)

    for chunk_start in range(start_row, total_rows, chunk_rows):
        chunk_end = min(chunk_start + chunk_rows, total_rows)
        chunk_index = chunk_start // chunk_rows
        seizure_chunk = seizures.iloc[chunk_start:chunk_end][needed_columns].copy()

        working_chunk, chunk_summary = _prepare_stereotyped_duration_chunk(
            seizure_chunk,
            birth_lookup,
            max_duration_seconds,
        )
        grouped_stats = None
        if not working_chunk.empty:
            grouped_stats = (
                working_chunk.groupby(['age_group', 'Unlinked_ID', 'type'], observed=True)['duration']
                .agg(count='count', mean='mean', var='var')
                .reset_index()
            )
            _merge_grouped_stereotyped_duration_stats(stats_by_key, grouped_stats)

        progress['processed_chunk_count'] = chunk_index + 1
        progress['source_rows_processed'] = int(progress.get('source_rows_processed', 0)) + chunk_summary['source_rows']
        progress['valid_rows_aggregated'] = int(progress.get('valid_rows_aggregated', 0)) + chunk_summary['retained_rows']
        progress['invalid_datetime_rows'] = int(progress.get('invalid_datetime_rows', 0)) + chunk_summary['invalid_datetime_rows']
        progress['over_duration_rows'] = int(progress.get('over_duration_rows', 0)) + chunk_summary['over_duration_rows']
        progress['last_completed_stage'] = 'aggregate_chunk'

        _write_pickle_atomic(stats_state_path, stats_by_key)
        _write_json_atomic(progress_path, progress)

        print(
            f"  completed chunk {chunk_index + 1}: kept {chunk_summary['retained_rows']:,} "
            f"of {chunk_summary['source_rows']:,} seizure rows"
        )

        del seizure_chunk
        del working_chunk
        if grouped_stats is not None:
            del grouped_stats
        gc.collect()

    del seizures
    gc.collect()

    progress['aggregation_complete'] = True
    progress['last_completed_stage'] = 'aggregate'
    _write_pickle_atomic(stats_state_path, stats_by_key)
    _write_json_atomic(progress_path, progress)
    return stats_by_key, progress


def _build_stereotyped_duration_arrays(stats_by_key):
    child_sds_minutes = []
    adult_sds_minutes = []

    for (age_group, _, _), (count, _, m2) in stats_by_key.items():
        if count < 3:
            continue

        sample_variance = m2 / (count - 1)
        if not np.isfinite(sample_variance):
            continue

        sd_minutes = np.sqrt(max(sample_variance, 0.0)) / 60.0
        if not np.isfinite(sd_minutes):
            continue

        if age_group == 'child':
            child_sds_minutes.append(sd_minutes)
        elif age_group == 'adult':
            adult_sds_minutes.append(sd_minutes)

    return (
        np.asarray(child_sds_minutes, dtype=float),
        np.asarray(adult_sds_minutes, dtype=float),
    )


def _plot_stereotyped_duration_histograms(child_sds, adult_sds, bins, plot_path, show_plot):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    plot_specs = [
        (child_sds, 'Children', '#2563eb', axes[0, 0], axes[1, 0]),
        (adult_sds, 'Adults', '#dc2626', axes[0, 1], axes[1, 1]),
    ]

    if bins == 'auto':
        full_range_bin_count = 50
        zoom_bin_count = 40
    elif isinstance(bins, int):
        full_range_bin_count = max(20, int(bins))
        zoom_bin_count = max(15, min(int(bins), 80))
    else:
        full_range_bin_count = 50
        zoom_bin_count = 40

    for sd_values, title, color, full_axis, zoom_axis in plot_specs:
        finite_values = np.asarray(sd_values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]

        for axis in (full_axis, zoom_axis):
            axis.grid(True, linestyle='--', alpha=0.35)

        if len(finite_values) == 0:
            for axis in (full_axis, zoom_axis):
                axis.text(
                    0.5,
                    0.5,
                    'No SDs computed\n(requires >=3 seizures per type)',
                    transform=axis.transAxes,
                    ha='center',
                    va='center',
                )
            continue

        zero_count = int(np.sum(finite_values == 0))
        positive_values = finite_values[finite_values > 0]

        median_sd = float(np.median(finite_values))
        p90_sd = float(np.percentile(finite_values, 90))
        p95_sd = float(np.percentile(finite_values, 95))
        p99_sd = float(np.percentile(finite_values, 99))
        q1_sd = float(np.percentile(finite_values, 25))
        q3_sd = float(np.percentile(finite_values, 75))
        max_sd = float(np.max(finite_values))

        full_axis.set_title(f'{title} (n={len(finite_values):,})')
        full_axis.set_ylabel('Groups (%)')
        full_axis.set_xlabel('SD (minutes, log scale)')

        if len(positive_values) > 0:
            min_positive = max(float(np.min(positive_values)), 1e-3)
            max_positive = float(np.max(positive_values))
            if min_positive == max_positive:
                log_bins = np.array([min_positive * 0.8, max_positive * 1.2])
            else:
                log_bins = np.geomspace(min_positive, max_positive, full_range_bin_count)

            full_axis.hist(
                positive_values,
                bins=log_bins,
                weights=np.full(len(positive_values), 100.0 / len(finite_values)),
                color=color,
                alpha=0.75,
                edgecolor='white',
                linewidth=0.5,
            )
            full_axis.set_xscale('log')

            if median_sd > 0:
                full_axis.axvline(median_sd, color=color, linewidth=2, label=f'median {median_sd:.2f}')
            if p95_sd > 0:
                full_axis.axvline(p95_sd, color=color, linestyle='--', linewidth=1.6, label=f'p95 {p95_sd:.1f}')

        full_axis.text(
            0.98,
            0.97,
            (
                f'Q1-Q3: {q1_sd:.2f}-{q3_sd:.2f} min\n'
                f'zero SD groups: {zero_count:,}\n'
                f'max: {max_sd:.1f} min'
            ),
            transform=full_axis.transAxes,
            ha='right',
            va='top',
            bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white', 'alpha': 0.9, 'edgecolor': color},
        )
        if len(positive_values) > 0:
            full_axis.legend(loc='upper left', frameon=True)

        zoom_limit = p99_sd if p99_sd > 0 else max_sd
        clipped_values = finite_values[finite_values <= zoom_limit]
        tail_count = int(len(finite_values) - len(clipped_values))

        zoom_axis.hist(
            clipped_values,
            bins=np.linspace(0, zoom_limit, zoom_bin_count),
            weights=np.full(len(clipped_values), 100.0 / len(finite_values)),
            color=color,
            alpha=0.75,
            edgecolor='white',
            linewidth=0.5,
        )
        zoom_axis.axvline(median_sd, color=color, linewidth=2)
        zoom_axis.axvline(p90_sd, color=color, linestyle='--', linewidth=1.6)
        zoom_axis.set_xlim(0, zoom_limit)
        zoom_axis.set_xlabel(f'SD (minutes, zoomed to p99={p99_sd:.1f})')
        zoom_axis.set_ylabel('Groups (%)')
        zoom_axis.text(
            0.98,
            0.97,
            f'median: {median_sd:.2f} min\np90: {p90_sd:.1f} min\noutside zoom: {tail_count:,}',
            transform=zoom_axis.transAxes,
            ha='right',
            va='top',
            bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white', 'alpha': 0.9, 'edgecolor': color},
        )

    fig.suptitle('Per-patient/per-type seizure duration variability', fontsize=14)
    fig.savefig(plot_path, dpi=300)
    if show_plot:
        plt.show()
    plt.close(fig)


def check_for_stereotyped_durations(
    do_raw_read=False,
    st_csv_path='STFullExportBIDMC_20240717.csv',
    profiles_pkl_path='profiles.pkl',
    seizures_pkl_path='seizures.pkl',
    bins='auto',
    run_parallel=False,
    stage_dir=None,
    chunk_rows=250000,
    resume=True,
    show_plot=True,
    max_duration_minutes=_STEREOTYPED_DURATION_DEFAULT_MAX_MINUTES,
):
    if run_parallel:
        print('run_parallel is disabled for low-memory staged processing.')

    if do_raw_read:
        resume = False

    csv_path = Path(st_csv_path).expanduser()
    profiles_pickle_path = Path(profiles_pkl_path).expanduser()
    seizures_pickle_path = Path(seizures_pkl_path).expanduser()

    if csv_path.exists():
        source_kind = 'csv'
    elif profiles_pickle_path.exists() and seizures_pickle_path.exists():
        source_kind = 'pickle'
    else:
        raise FileNotFoundError(
            'No usable source found for stereotyped-duration analysis. '
            'Expected the raw export CSV or both profiles/seizures pickles.'
        )

    stage_dir_path = _resolve_stereotyped_duration_stage_dir(stage_dir, st_csv_path)
    progress_path = stage_dir_path / 'progress.json'
    birth_lookup_path = stage_dir_path / 'profiles_birth_lookup.pkl'
    stats_state_path = stage_dir_path / 'stats_state.pkl'
    results_path = stage_dir_path / 'stereotyped_duration_results.npz'
    summary_path = stage_dir_path / 'summary.json'
    plot_path = stage_dir_path / 'stereotyped_duration_hist.png'
    max_duration_seconds = None if max_duration_minutes is None else float(max_duration_minutes) * 60.0

    source_signature = _build_stereotyped_duration_source_signature(
        source_kind=source_kind,
        st_csv_path=st_csv_path,
        profiles_pkl_path=profiles_pkl_path,
        seizures_pkl_path=seizures_pkl_path,
        chunk_rows=chunk_rows,
        max_duration_minutes=max_duration_minutes,
    )
    progress = _initialize_stereotyped_duration_stage_dir(
        stage_dir=stage_dir_path,
        source_kind=source_kind,
        source_signature=source_signature,
        chunk_rows=chunk_rows,
        resume=resume,
    )

    print(f'Using {source_kind} source for staged stereotyped-duration analysis.')
    print(f'Checkpoint directory: {stage_dir_path}')
    if max_duration_minutes is None:
        print('No seizure-duration cutoff applied.')
    else:
        print(f'Ignoring seizure durations longer than {float(max_duration_minutes):.0f} minutes.')

    if not progress.get('profiles_ready') or not birth_lookup_path.exists():
        if source_kind == 'csv':
            print('Stage 1/3: building birth-date lookup from raw Profiles section...')
            birth_lookup = _extract_profiles_birth_lookup_from_csv(csv_path)
        else:
            print('Stage 1/3: building birth-date lookup from profiles pickle...')
            birth_lookup = _extract_profiles_birth_lookup_from_pickle(profiles_pickle_path)

        _write_pickle_atomic(birth_lookup_path, birth_lookup)
        progress['profiles_ready'] = True
        progress['last_completed_stage'] = 'profiles'
        _write_json_atomic(progress_path, progress)
        del birth_lookup
        gc.collect()
    else:
        print('Stage 1/3: reusing saved birth-date lookup.')

    birth_lookup = _read_pickle_if_exists(birth_lookup_path)
    if not isinstance(birth_lookup, dict) or not birth_lookup:
        raise ValueError('Birth-date lookup checkpoint is missing or invalid')

    if not progress.get('aggregation_complete') or not stats_state_path.exists():
        resume_chunk = int(progress.get('processed_chunk_count', 0))
        print(f'Stage 2/3: aggregating seizure statistics (resuming at chunk {resume_chunk})...')
        if source_kind == 'csv':
            stats_by_key, progress = _aggregate_stereotyped_duration_stats_from_csv(
                st_csv_path=csv_path,
                birth_lookup=birth_lookup,
                stage_dir=stage_dir_path,
                progress=progress,
                chunk_rows=chunk_rows,
                max_duration_seconds=max_duration_seconds,
            )
        else:
            stats_by_key, progress = _aggregate_stereotyped_duration_stats_from_pickle(
                seizures_pkl_path=seizures_pickle_path,
                birth_lookup=birth_lookup,
                stage_dir=stage_dir_path,
                progress=progress,
                chunk_rows=chunk_rows,
                max_duration_seconds=max_duration_seconds,
            )
    else:
        print('Stage 2/3: reusing saved aggregated statistics.')
        stats_by_key = _read_pickle_if_exists(stats_state_path)
        if not isinstance(stats_by_key, dict):
            raise ValueError('Aggregated stereotyped-duration checkpoint is missing or invalid')

    if not progress.get('final_ready') or not results_path.exists() or not summary_path.exists():
        print('Stage 3/3: finalizing result arrays and summary...')
        child_sds, adult_sds = _build_stereotyped_duration_arrays(stats_by_key)

        child_median = float(np.median(child_sds)) if len(child_sds) else np.nan
        adult_median = float(np.median(adult_sds)) if len(adult_sds) else np.nan
        _save_npz_atomic(
            results_path,
            child_sds_minutes=child_sds,
            adult_sds_minutes=adult_sds,
        )
        summary = {
            'child_group_count': int(len(child_sds)),
            'adult_group_count': int(len(adult_sds)),
            'child_median_sd_minutes': None if np.isnan(child_median) else float(child_median),
            'adult_median_sd_minutes': None if np.isnan(adult_median) else float(adult_median),
            'processed_chunk_count': int(progress.get('processed_chunk_count', 0)),
            'source_rows_processed': int(progress.get('source_rows_processed', 0)),
            'valid_rows_aggregated': int(progress.get('valid_rows_aggregated', 0)),
            'invalid_datetime_rows': int(progress.get('invalid_datetime_rows', 0)),
            'over_duration_rows': int(progress.get('over_duration_rows', 0)),
            'max_duration_minutes': None if max_duration_minutes is None else float(max_duration_minutes),
        }
        _write_json_atomic(summary_path, summary)
        progress['final_ready'] = True
        progress['last_completed_stage'] = 'finalize'
        _write_json_atomic(progress_path, progress)
    else:
        print('Stage 3/3: reusing saved result arrays.')
        with np.load(results_path) as result_file:
            child_sds = result_file['child_sds_minutes']
            adult_sds = result_file['adult_sds_minutes']
        summary = _read_json_if_exists(summary_path) or {}
        child_median = float(np.median(child_sds)) if len(child_sds) else np.nan
        adult_median = float(np.median(adult_sds)) if len(adult_sds) else np.nan

    _plot_stereotyped_duration_histograms(
        child_sds=child_sds,
        adult_sds=adult_sds,
        bins=bins,
        plot_path=plot_path,
        show_plot=show_plot,
    )

    print(
        f"Completed stereotyped-duration analysis with {len(child_sds):,} child groups and "
        f"{len(adult_sds):,} adult groups."
    )

    del birth_lookup
    del stats_by_key
    gc.collect()

    return {
        'child_sds_minutes': child_sds,
        'adult_sds_minutes': adult_sds,
        'child_median_sd_minutes': child_median,
        'adult_median_sd_minutes': adult_median,
        'max_duration_minutes': None if max_duration_minutes is None else float(max_duration_minutes),
        'stage_dir': str(stage_dir_path),
        'summary_path': str(summary_path),
        'plot_path': str(plot_path),
        'source_kind': source_kind,
    }

def check_SF(do_raw_read=False):
    # Read the SeizureTracker CSV file directly, or just a pickle of it already read in
    if do_raw_read:
        #profiles, seizures = read_seizuretracker_csv('STFullExportBIDMC_20220808_A.csv')
        profiles, seizures = read_seizuretracker_csv('STFullExportBIDMC_20240717.csv')

        print('Saving pickles...',end='')
        profiles.to_pickle('profiles.pkl')
        seizures.to_pickle('seizures.pkl')
        print('done.')
    else:
        print('Loading pickles...',end='')
        profiles = pd.read_pickle('profiles.pkl')
        seizures = pd.read_pickle('seizures.pkl')
        print('done.')

    # calculate the seizure frequency per patient
    # use the first seizure date and the last seizure date to get the duration
    # NOTE: patients with only one seizure (or identical min/max timestamps) have duration_days == 0,
    # which would otherwise yield Inf seizure_frequency.
    seizure_dates = seizures.groupby('Unlinked_ID')['Date_Time'].agg(['min', 'max', 'count']).reset_index()
    seizure_dates['Date_Time_min'] = pd.to_datetime(seizure_dates['min'], errors='coerce')
    seizure_dates['Date_Time_max'] = pd.to_datetime(seizure_dates['max'], errors='coerce')
    seizure_dates['duration_days'] = (
        (seizure_dates['Date_Time_max'] - seizure_dates['Date_Time_min']).dt.total_seconds()
        / (24 * 60 * 60)
    )
    seizure_dates.loc[seizure_dates['duration_days'] <= 0, 'duration_days'] = np.nan
    seizure_dates['seizure_frequency'] = 30 * (seizure_dates['count'] / seizure_dates['duration_days'])

    # compute correlation between seizure frequency and median seizure duration per Unlinked_ID
    seizure_frequencies = seizure_dates[['Unlinked_ID', 'seizure_frequency']]
    
    seizure_durations = seizures[seizures['duration'] > 0].groupby('Unlinked_ID')['duration'].median().reset_index(name='median_duration')
    seizure_stats = pd.merge(seizure_frequencies, seizure_durations, on='Unlinked_ID', how='inner')
    # error checking: make sure only valid seizure frequencies and durations are included
    seizure_stats = seizure_stats[(seizure_stats['seizure_frequency'] > 0) & (seizure_stats['median_duration'] > 0)].copy()
    seizure_stats = seizure_stats[
        np.isfinite(seizure_stats['seizure_frequency'].to_numpy())
        & np.isfinite(seizure_stats['median_duration'].to_numpy())
    ]

    if len(seizure_stats) == 0:
        print('No valid points for seizure frequency vs duration (empty after filtering).')
        return

    if len(seizure_stats) < 2:
        print('Not enough valid points to compute correlation (need >= 2).')
    else:
        correlation = seizure_stats['seizure_frequency'].corr(seizure_stats['median_duration'])
        if pd.isna(correlation):
            print('Correlation between seizure frequency and median seizure duration: NaN (insufficient variance).')
        else:
            print(f'Correlation between seizure frequency and median seizure duration: {correlation:.4f}')
    # plot a scatterplot of seizure frequency vs median seizure duration
    plt.figure(figsize=(6,6))
    plt.scatter(seizure_stats['seizure_frequency'], seizure_stats['median_duration']/60, alpha=0.5)
    plt.xlabel('Seizure Frequency (seizures per month)')
    plt.ylabel('Median Seizure Duration (minutes)')
    plt.title('Seizure Frequency vs Median Seizure Duration')
    plt.grid(True, linestyle='--', alpha=0.7)
    x_max = seizure_stats['seizure_frequency'].max()
    if pd.notna(x_max) and np.isfinite(x_max) and x_max > 0:
        plt.xlim(0, x_max * 1.1)

    y_max = (seizure_stats['median_duration'] / 60).max()
    if pd.notna(y_max) and np.isfinite(y_max) and y_max > 0:
        plt.ylim(0, y_max * 1.1)
    plt.show()


def check_gender(do_raw_read=False):
    # Read the SeizureTracker CSV file directly, or just a pickle of it already read in
    if do_raw_read:
        #profiles, seizures = read_seizuretracker_csv('STFullExportBIDMC_20220808_A.csv')
        profiles, seizures = read_seizuretracker_csv('STFullExportBIDMC_20240717.csv')

        print('Saving pickles...',end='')
        profiles.to_pickle('profiles.pkl')
        seizures.to_pickle('seizures.pkl')
        print('done.')
    else:
        print('Loading pickles...',end='')
        profiles = pd.read_pickle('profiles.pkl')
        seizures = pd.read_pickle('seizures.pkl')
        print('done.')

    output_file = 'CDF_gender.csv'
    # Merge with profiles to get gender
    merged = pd.merge(seizures, profiles[['Unlinked_ID', 'Gender']], on='Unlinked_ID', how='left').copy()
    # remove any entry without a seizure
    merged = merged[merged['duration'] > 0].copy()

    gender_norm = merged['Gender'].astype('string').str.strip().str.lower()
    gender_norm = gender_norm.replace({
        'f': 'female',
        'm': 'male',
        'woman': 'female',
        'man': 'male',
        'girl': 'female',
        'boy': 'male',
    })
    merged['Gender_norm'] = gender_norm

    plt.figure(figsize=(6,6))
    plt.subplot(1,2,1)
    thetype = 'all'
    drawGraph(merged[merged['Gender_norm'] == 'female'],agetype='',as_subplots=True,thetype=thetype,output_file=output_file)
    plt.title('All females')
    plt.xlabel('') # remove xlabel
    plt.subplot(1,2,2)
    drawGraph(merged[merged['Gender_norm'] == 'male'],agetype='',as_subplots=True,thetype=thetype,output_file=output_file)
    plt.title('All males')
    plt.xlabel('') # remove xlabel

def do_full_display(do_raw_read=False):

    # Read the SeizureTracker CSV file directly, or just a pickle of it already read in
    if do_raw_read:
        #profiles, seizures = read_seizuretracker_csv('STFullExportBIDMC_20220808_A.csv')
        profiles, seizures = read_seizuretracker_csv('STFullExportBIDMC_20240717.csv')

        print('Saving pickles...',end='')
        profiles.to_pickle('profiles.pkl')
        seizures.to_pickle('seizures.pkl')
        print('done.')
    else:
        print('Loading pickles...',end='')
        profiles = pd.read_pickle('profiles.pkl')
        seizures = pd.read_pickle('seizures.pkl')
        print('done.')

    # Replace 'nan' with 'Unknown*' in the 'type' column
    seizures['type'] = seizures['type'].fillna('Unknown*')
    # Replace 'Desconocido' with 'Unknown*' in the 'type' column
    seizures['type'] = seizures['type'].replace('Desconocido', 'Unknown*')


    csv_child = 'table1_child.csv'
    csv_adult = 'table1_adult.csv'
    output_file = 'CDF_data.csv'

    child_seizures, adult_seizures = age_cut(seizures, profiles)
    # child seizures first
    make_table_1(child_seizures, profiles,csv_child)
    make_table_1(adult_seizures, profiles,csv_adult)

    
    #  first

    txtcols = 'cdftype,type,agetype,nseizures,npatients,prob90_time,median'
    with open(output_file, 'w') as file:
        file.write(txtcols + '\n')

    # Create child_seizures_GTC with GTC combined types
    child_seizures_GTC = child_seizures.copy()
    child_seizures_GTC['type'] = child_seizures_GTC['type'].replace({
        'Focal to bilateral tonic clonic*': 'GTC',
        'Tonic Clonic':                    'GTC'
    })
    # Create adult_seizures_GTC with GTC combined types
    adult_seizures_GTC = adult_seizures.copy()
    adult_seizures_GTC['type'] = adult_seizures_GTC['type'].replace({
        'Focal to bilateral tonic clonic*': 'GTC',
        'Tonic Clonic':                    'GTC'
    })

    # Create child_seizures_IA with all unaware types
    #IA = 'Impaired aware'
    IA = 'Impaired Consciousness'
    child_seizures_IA = child_seizures.copy()
    child_seizures_IA['type'] = child_seizures_IA['type'].replace({
        'Absence': IA,
        'Atypical Absence': IA,
        'Focal to bilateral tonic clonic*': IA,
        'Focal impaired awareness*': IA,
        'Tonic': IA,
        'Clonic': IA,
        'Gelastic': IA,
        'Tonic Clonic': IA
    })
    # Create adult_seizures_GTC with GTC combined types
    adult_seizures_IA = adult_seizures.copy()
    adult_seizures_IA['type'] = adult_seizures_GTC['type'].replace({
        'Absence': IA,
        'Atypical Absence': IA,
        'Focal to bilateral tonic clonic*': IA,
        'Focal impaired awareness*': IA,
        'Tonic': IA,
        'Clonic': IA,
        'Gelastic': IA,
        'Tonic Clonic': IA
    })
    
    # build a special FIGURE 1 with 4 parts

    fname = 'Figure1-comboFig.png'
    output_file = 'Figure1-combofile.csv'
    plt.figure(figsize=(6,6))
    ax11 = plt.subplot(2,2,1)
    thetype = 'all'
    drawGraph(child_seizures,limx=20,agetype='child',as_subplots=True,thetype=thetype,output_file=output_file)
    drawGraph(adult_seizures,limx=20,agetype='adult',as_subplots=True,thetype=thetype,output_file=output_file)
    plt.xlim(0,20)
    ax = plt.gca()  # Get the current Axes instance
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))  # Set minor ticks for x-axis, between the major ticks
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))  # Set minor ticks for y-axis, between the major ticks
    # Add minor grid lines
    ax.grid(True, linestyle=':', alpha=0.5, which='minor')  # Minor grid lines
    #plt.xticks([])  # Remove x-axis tick labels
    plt.ylabel('') # remove ylabel
    plt.legend()
    ax11.set_title('All by seizures')
    plt.xlabel('') # remove xlabel
    ax12 = plt.subplot(2,2,2)
    thetype = 'all'
    xlimlist=(10,20)
    drawGraph_population(child_seizures,agetype='child',as_subplots=True,limx= xlimlist[0], thetype=thetype,output_file=output_file)
    drawGraph_population(adult_seizures,agetype='adult',as_subplots=True,limx= xlimlist[1], thetype=thetype,output_file=output_file)
    ax12.set_title('All by patient')
    plt.xlabel('') # remove xlabel
    plt.ylabel('') # remove ylabel
    #plt.xticks([])  # Remove x-axis tick labels
    ax21 = plt.subplot(2,2,3)
    thetype = 'GTC'
    drawGraph_population(child_seizures_GTC,agetype='child',as_subplots=True,limx= xlimlist[0], thetype=thetype,output_file=output_file)
    drawGraph_population(adult_seizures_GTC,agetype='adult',as_subplots=True,limx= xlimlist[1], thetype=thetype,output_file=output_file)
    ax21.set_title('GTC+FBTC by patient')
    ax22 = plt.subplot(2,2,4)
    thetype = IA
    drawGraph_population(child_seizures_IA,agetype='child',as_subplots=True,limx= xlimlist[0], thetype=thetype,output_file=output_file)
    drawGraph_population(adult_seizures_IA,agetype='adult',as_subplots=True,limx= xlimlist[1], thetype=thetype,output_file=output_file)
    #plt.title('Focal impaired consciousness by patient')
    ax22.set_title('Focal impaired\nconsciousness by patient')
    plt.ylabel('') # remove ylabel
    #plt.xlabel('') # remove xlabel
    #plt.xticks([])  # Remove x-axis tick labels
    plt.tight_layout(rect=(0.02, 0.02, 1, 0.98))
    # Add bold panel labels after tight_layout so they are not repositioned/clipped
    for ax, label in zip([ax11, ax12, ax21, ax22], ['A', 'B', 'C', 'D']):
        title_obj = ax.title
        ax.text(
            -0.02, 1.02, label,
            transform=ax.transAxes,
            fontsize=title_obj.get_fontsize() + 2,
            fontweight='bold',
            va='bottom', ha='right',
            clip_on=False,
        )
    plt.savefig(fname,dpi=300)
    plt.show()

    # this is mainly just for the output file.
    thetype = 'GTC'
    drawGraph(child_seizures_GTC,agetype='child',as_subplots=True,limx= xlimlist[0], thetype=thetype,output_file=output_file)
    drawGraph(adult_seizures_GTC,agetype='adult',as_subplots=True,limx= xlimlist[1], thetype=thetype,output_file=output_file)
    plt.show()
    thetype = IA
    drawGraph(child_seizures_IA,agetype='child',as_subplots=True,limx= xlimlist[0], thetype=thetype,output_file=output_file)
    drawGraph(adult_seizures_IA,agetype='adult',as_subplots=True,limx= xlimlist[1], thetype=thetype,output_file=output_file)
    plt.show()


    make_a_pair_of_CDFs(child_seizures, adult_seizures, thetype='all',titlestr='All types',fname='Fig1-CDF-all.png',output_file=output_file,xlimlist=(10,25))

   
    make_a_pair_of_CDFs(child_seizures_GTC, adult_seizures_GTC, thetype='GTC',titlestr='GTC + FBTC',fname='Fig1C-CDF-GTC.png',output_file=output_file,xlimlist=(10,25))


    for thetype in seizures['type'].unique():
        print(f'Processing {thetype}...')
        make_a_pair_of_CDFs(child_seizures, adult_seizures, thetype=thetype,titlestr=thetype,fname=f'Fig1-CDF-{thetype}.png',output_file=output_file)


    # Create a figure with four subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    # Child seizures
    res_child = analyze_seizure_durations(child_seizures)
    res_all_child = analyze_durations_all(child_seizures)
    compare_results(res_child, res_all_child, ax1, ax2, agetype='Child')

    # Adult seizures
    res_adult = analyze_seizure_durations(adult_seizures)
    res_all_adult = analyze_durations_all(adult_seizures)
    compare_results(res_adult, res_all_adult, ax3, ax4, agetype='Adult')

    plt.tight_layout()
    plt.savefig('APPENDOX-fig1-compare.png',dpi=300)
    plt.show()



    # make boxplots third
    plt.figure(figsize=(12, 7))
    make_boxplot(child_seizures, adult_seizures)
    plt.savefig('Fig2-boxplot.png',dpi=300)
    plt.show()

    if 0:
        drawSimplerBoxes(res_child, res_adult)
        makeBoxPlotsV2(child_seizures, adult_seizures)
    #plt.figure(figsize=(7, 10))
    #plt.subplot(2,1,1)
    #bigCDF(child_seizures, max_duration=None,agetype='child',as_subplots=True)
    #plt.subplot(2,1,2)
    #bigCDF(adult_seizures, max_duration=None,agetype='adult',as_subplots=True)
    #plt.show()


# Usage
#file_path = 'path/to/your/seizuretracker_file.csv'
#profiles, seizures = read_seizuretracker_csv(file_path)

#print("Profiles DataFrame:")
#print(profiles.head())
#print("\nSeizures DataFrame:")
#print(seizures.head())